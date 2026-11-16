"""9장 예제 — 100B인데 5W? MoE와 메모리 안 계산(PIM)을 숫자로
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-09-moe-pim/

python example.py   (CPU, 다운로드 없음, 약 1분. 메모리 ~3GB 사용)
"""
import sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from common.util import setup, save
from common.roofline import roofline_calc, measure_bw, timeit, EDGE_NPU

# 공개 config.json (Hugging Face) 값
GPT_OSS_20B = dict(name="gpt-oss-20b", L=24, d=2880, ffn=2880, E=32, k=4, H=64, Hkv=8, hd=64, vocab=201088, tied=False)
MIXTRAL = dict(name="Mixtral-8x7B", L=32, d=4096, ffn=14336, E=8, k=2, H=32, Hkv=8, hd=128, vocab=32000, tied=False)


def moe_params(c):
    expert = 3 * c["d"] * c["ffn"]
    attn = c["d"] * c["H"] * c["hd"] * 2 + c["d"] * c["Hkv"] * c["hd"] * 2
    emb = c["vocab"] * c["d"]
    total = c["L"] * (c["E"] * expert + attn + c["E"] * c["d"]) + emb * (1 if c["tied"] else 2)
    active = c["L"] * (c["k"] * expert + attn + c["E"] * c["d"]) + emb     # 임베딩은 1행만 읽으니 unembedding만
    return dict(total=total, active=active, experts_active=c["L"] * c["k"] * expert,
                attn=c["L"] * attn, unembed=emb)


def calc(R):
    out = {}
    for c in (GPT_OSS_20B, MIXTRAL):
        p = moe_params(c); out[c["name"]] = p
        print(f"[1] {c['name']}: 총 {p['total']/1e9:.2f}B, 활성 {p['active']/1e9:.2f}B "
              f"(전문가 {p['experts_active']/1e9:.2f}B + 어텐션 {p['attn']/1e9:.2f}B + unembedding {p['unembed']/1e9:.2f}B)")
    g = out["gpt-oss-20b"]
    # 공개 체크포인트: 전문가 MXFP4(4bit + 32개당 8bit scale = 4.25bit), 어텐션·unembedding BF16
    b_ckpt = g["experts_active"] * 4.25 / 8 + (g["attn"] + g["unembed"]) * 2 + 24 * 32 * 2880 * 2
    b_all4 = g["active"] * 4.25 / 8
    bw = EDGE_NPU["bw"]
    rows = {"활성 3.6B 전부 4.25bit (가정)": bw / b_all4,
            "공개 체크포인트 형식 그대로 (전문가 MXFP4, 나머지 BF16)": bw / b_ckpt,
            "밀집 21B 전부 4.25bit": bw / (g["total"] * 4.25 / 8)}
    for k, v in rows.items():
        print(f"    가상 엣지 NPU 153.6 GB/s 상한 — {k}: {v:.1f} tok/s")
    out["gptoss_bounds"] = rows; out["gptoss_bytes"] = dict(ckpt=b_ckpt, all4=b_all4)
    # 메모리 용량: 활성이 아니라 '총' 파라미터가 들어가야 함
    out["capacity_GB"] = {c["name"]: moe_params(c)["total"] * 4.25 / 8 / 1e9 for c in (GPT_OSS_20B, MIXTRAL)}
    print("    용량(전부 4.25bit 가정):", {k: f"{v:.1f} GB" for k, v in out["capacity_GB"].items()})
    R["calc"] = out


# ---------- 미니 MoE 층
class MoE(torch.nn.Module):
    def __init__(self, d, ffn, E, k, seed=0, bias=None):
        super().__init__(); torch.manual_seed(seed)
        self.E, self.k = E, k
        self.router = torch.nn.Linear(d, E, bias=False)
        self.w1 = torch.randn(E, ffn, d) * 0.02; self.w3 = torch.randn(E, ffn, d) * 0.02
        self.w2 = torch.randn(E, d, ffn) * 0.02
        self.bias = bias if bias is not None else torch.zeros(E)
        self.hits = torch.zeros(E, dtype=torch.long)

    def forward(self, x):                               # x: (T, d)
        logits = self.router(x) + self.bias
        w, idx = logits.topk(self.k, -1); w = w.softmax(-1)
        y = torch.zeros_like(x)
        for e in idx.unique().tolist():                 # 고른 전문가만 읽는다
            tok, slot = (idx == e).nonzero(as_tuple=True); self.hits[e] += len(tok)
            h = F.silu(x[tok] @ self.w1[e].T) * (x[tok] @ self.w3[e].T)
            y[tok] += w[tok, slot, None] * (h @ self.w2[e].T)
        return y, idx


def measure(R):
    d, ffn, E, k, L = 1024, 2816, 16, 2, 4
    layers = [MoE(d, ffn, E, k, seed=i) for i in range(L)]
    dense = [(torch.randn(E * ffn, d) * 0.02, torch.randn(E * ffn, d) * 0.02, torch.randn(d, E * ffn) * 0.02)
             for _ in range(L)]                         # 총 파라미터가 같은 밀집 FFN
    exp_bytes = 3 * d * ffn * 4
    bw = measure_bw()
    print(f"\n[2] 미니 MoE: {L}층 × 전문가 {E}개(top-{k}), 전문가 1개 {exp_bytes/1e6:.1f} MB, 총 {L*E*exp_bytes/1e9:.2f} GB (FP32), 측정 BW {bw/1e9:.1f} GB/s")
    g = torch.Generator().manual_seed(1)
    def run_moe(T):
        x = torch.randn(T, d, generator=g)
        for l in layers: x = x + l(x)[0]
    def run_dense(T):
        x = torch.randn(T, d, generator=g)
        for w1, w3, w2 in dense: x = x + (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    res = []
    with torch.no_grad():
        for T in (1, 2, 4, 8, 16, 32):
            for l in layers: l.hits.zero_()
            tm = timeit(lambda: run_moe(T), reps=5, warmup=1)
            uniq = [int((l.hits > 0).sum()) for l in layers]
            td = timeit(lambda: run_dense(T), reps=3, warmup=1)
            pred_distinct = E * (1 - (1 - k / E) ** T)         # 균등 라우팅 가정
            meas_distinct = sum(uniq) / L / 6                   # 1+5회 실행 평균 아님 → 아래에서 따로 셈
            res.append(dict(T=T, moe_ms=tm * 1e3, dense_ms=td * 1e3, pred_distinct=pred_distinct,
                            pred_moe_ms=L * pred_distinct * exp_bytes / bw * 1e3, pred_dense_ms=L * E * exp_bytes / bw * 1e3))
    # 서로 다른 전문가 수: 한 번 실행에서 직접 센다
    with torch.no_grad():
        for r in res:
            x = torch.randn(r["T"], d, generator=g); cnt = []
            for l in layers:
                _, idx = l(x); cnt.append(len(idx.unique()))
            r["meas_distinct"] = sum(cnt) / L
            print(f"  토큰 {r['T']:3d}개: 서로 다른 전문가 실측 {r['meas_distinct']:5.2f} / 식 {r['pred_distinct']:5.2f}   "
                  f"MoE {r['moe_ms']:7.1f} ms (예측 {r['pred_moe_ms']:6.1f})   밀집(총량 같음) {r['dense_ms']:7.1f} ms (예측 {r['pred_dense_ms']:6.1f})")
    R["measure"] = res; R["bw"] = bw
    del dense


def imbalance(R):
    # 라우터 편향 → 부하 불균형, 전문가 캐시(LRU) 적중률
    E, k, T = 16, 2, 4096
    g = torch.Generator().manual_seed(2); out = {}
    for name, bias in (("균형", torch.zeros(E)), ("편향(상위 2개 +2.0)", torch.tensor([2.0, 2.0] + [0.0] * 14))):
        m = MoE(64, 16, E, k, bias=bias)
        with torch.no_grad():
            _, idx = m(torch.randn(T, 64, generator=g))
        cnt = torch.bincount(idx.flatten(), minlength=E).float()
        frac = (cnt / cnt.sum()).tolist()
        out[name] = dict(frac=frac, max_over_mean=(cnt.max() / cnt.mean()).item())
        # LRU 캐시: 빠른 메모리에 전문가 C개만 둘 때
        for C in (4, 8, 12):
            cache, hits = [], 0
            for row in idx.tolist():
                for e in row:
                    if e in cache: hits += 1; cache.remove(e)
                    elif len(cache) >= C: cache.pop(0)
                    cache.append(e)
            out[name][f"lru{C}"] = hits / (T * k)
        print(f"[3] 라우팅 {name}: 가장 바쁜 전문가 = 평균의 {out[name]['max_over_mean']:.2f}배, "
              f"LRU 적중률 C=4/8/12: {out[name]['lru4']:.2f}/{out[name]['lru8']:.2f}/{out[name]['lru12']:.2f} (균등 예측 C/E = {4/E:.2f}/{8/E:.2f}/{12/E:.2f})")
    R["imbalance"] = out


def pim(R):
    # PIM: GEMV를 DRAM 뱅크 옆에서 하면 '내부' 대역폭이 외부 버스의 m배라고 가정(가정치)
    act = 3.6e9; bits = 4.25; ext = EDGE_NPU["bw"]; rows = {}
    for m in (1, 2, 4, 8, 16):
        # 가정: 가중치 GEMV(전체 바이트의 90%)는 PIM이 m배 대역폭으로, 나머지 10%(어텐션·KV 등)는 외부 버스로
        t = act * bits / 8 * (0.9 / (ext * m) + 0.1 / ext)
        rows[m] = 1 / t
    R["pim"] = rows
    print("[4] PIM 가정 계산 (GEMV 90%만 내부 대역폭 m배):", {m: f"{v:.0f} tok/s" for m, v in rows.items()})
    print(f"    m→∞여도 상한 {1/(act*bits/8*0.1/ext):.0f} tok/s — 나머지 10%가 Amdahl 벽(4장)")
    R["pim_inf"] = 1 / (act * bits / 8 * 0.1 / ext)


if __name__ == "__main__":
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    calc(R); measure(R); imbalance(R); pim(R)
    save(Path(__file__).parent, "09", R)
