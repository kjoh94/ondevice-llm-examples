"""공용 미니 Llama — 온디바이스 LLM 시리즈 3·4·5·7·8장이 같이 쓰는 모델.

RMSNorm · RoPE(사전계산 테이블) · GQA · SwiGLU · 정적(static) KV 캐시를 갖춘
200줄 이내의 디코더. 가중치는 시드로 고정한 랜덤 값(다운로드 없음).
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-series/
"""
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 2048
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 2          # GQA: Q 헤드 8개가 KV 헤드 2개를 나눠 씀
    ffn: int = 704
    max_seq: int = 512
    rope_theta: float = 10000.0
    eps: float = 1e-5

    @property
    def head_dim(self):
        return self.dim // self.n_heads


def n_params(cfg: Config) -> int:
    hd = cfg.head_dim
    attn = cfg.dim * cfg.dim * 2 + cfg.dim * cfg.n_kv_heads * hd * 2
    mlp = 3 * cfg.dim * cfg.ffn
    return cfg.n_layers * (attn + mlp + 2 * cfg.dim) + 2 * cfg.vocab * cfg.dim + cfg.dim


# ----------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim)); self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def rope_tables(cfg: Config):
    """cos/sin 테이블을 미리 계산(상수 폴딩 대상). shape (max_seq, head_dim/2)."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2).float() / cfg.head_dim))
    ang = torch.outer(torch.arange(cfg.max_seq).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd). 짝수/홀수 쌍을 2D 회전 — 원소별 연산."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


# ----------
class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg; hd = cfg.head_dim
        self.q = nn.Linear(cfg.dim, cfg.n_heads * hd, bias=False)
        self.k = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.v = nn.Linear(cfg.dim, cfg.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * hd, cfg.dim, bias=False)

    def forward(self, x, cos, sin, cache, pos, mask):
        B, T, _ = x.shape; c = self.cfg; hd = c.head_dim
        q = self.q(x).view(B, T, c.n_heads, hd).transpose(1, 2)
        k = self.k(x).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        v = self.v(x).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k, v = cache.update(k, v, pos)                  # 정적 캐시면 (B,Hkv,S,hd)
        rep = c.n_heads // c.n_kv_heads                 # GQA: KV 헤드를 복제해서 Q와 맞춤
        if rep > 1:
            k = k.repeat_interleave(rep, 1); v = v.repeat_interleave(rep, 1)
        att = (q @ k.transpose(-1, -2)) / hd ** 0.5 + mask
        y = att.softmax(-1) @ v
        return self.o(y.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.dim, cfg.eps); self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.dim, cfg.eps)
        self.gate = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn, bias=False)
        self.down = nn.Linear(cfg.ffn, cfg.dim, bias=False)

    def forward(self, x, cos, sin, cache, pos, mask):
        x = x + self.attn(self.n1(x), cos, sin, cache, pos, mask)
        h = self.n2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))   # SwiGLU


# ----------
class StaticCache:
    """MAXLEN 칸을 미리 잡아 두고 pos 위치에 써 넣는 캐시 → shape이 절대 안 바뀜."""
    def __init__(self, cfg, batch=1, dtype=torch.float32, sliced=False):
        shape = (batch, cfg.n_kv_heads, cfg.max_seq, cfg.head_dim)
        self.k = torch.zeros(shape, dtype=dtype); self.v = torch.zeros(shape, dtype=dtype)
        self.sliced = sliced        # True면 채워진 앞부분만 읽음(버퍼는 고정, 읽는 길이는 가변)

    def update(self, k, v, pos):
        T = k.shape[2]
        self.k[:, :, pos:pos + T] = k; self.v[:, :, pos:pos + T] = v
        if self.sliced:
            return self.k[:, :, :pos + T], self.v[:, :, :pos + T]
        return self.k, self.v


class DynamicCache:
    """torch.cat으로 늘려 가는 캐시 → 매 스텝 shape이 바뀜(그래프 컴파일러가 싫어함)."""
    def __init__(self):
        self.k = None; self.v = None

    def update(self, k, v, pos):
        self.k = k if self.k is None else torch.cat([self.k, k], 2)
        self.v = v if self.v is None else torch.cat([self.v, v], 2)
        return self.k, self.v


class LayerCaches(list):
    pass


def make_caches(cfg, kind="static", batch=1):
    if kind == "dynamic":
        return LayerCaches(DynamicCache() for _ in range(cfg.n_layers))
    return LayerCaches(StaticCache(cfg, batch, sliced=(kind == "sliced")) for _ in range(cfg.n_layers))


class TinyLlama(nn.Module):
    def __init__(self, cfg: Config, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.eps)
        self.head = nn.Linear(cfg.dim, cfg.vocab, bias=False)
        for p in self.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
        cos, sin = rope_tables(cfg)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, ids, pos: int, caches, position_ids=None):
        """ids: (B,T). pos: 캐시에 쓸 시작 위치. position_ids를 따로 주면 RoPE에 그걸 씀."""
        B, T = ids.shape
        if position_ids is None:
            position_ids = torch.arange(pos, pos + T)
        cos, sin = self.cos[position_ids], self.sin[position_ids]
        full = isinstance(caches[0], StaticCache) and not caches[0].sliced
        S = self.cfg.max_seq if full else pos + T
        # 쿼리 i(절대 위치 pos+i)는 키 j <= pos+i 만 본다. 정적 캐시의 빈칸(j > pos+T-1)도 가려짐.
        qpos = torch.arange(pos, pos + T).unsqueeze(1); kpos = torch.arange(S).unsqueeze(0)
        mask = torch.where(kpos <= qpos, 0.0, float("-inf"))
        x = self.emb(ids)
        for layer, cache in zip(self.layers, caches):
            x = layer(x, cos, sin, cache, pos, mask)
        return self.head(self.norm(x))


@torch.no_grad()
def greedy(model, prompt, n_new, kind="static"):
    """prefill 1번 + decode n_new-1번. 생성 토큰 리스트 반환."""
    caches = make_caches(model.cfg, kind, prompt.shape[0])
    logits = model(prompt, 0, caches)
    out = [int(logits[0, -1].argmax())]; pos = prompt.shape[1]
    for _ in range(n_new - 1):
        logits = model(torch.tensor([[out[-1]]]), pos, caches); pos += 1
        out.append(int(logits[0, -1].argmax()))
    return out
