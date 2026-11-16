"""3장 예제 — 메모장이 가중치보다 커질 때: KV 캐시 크기·예산·정적 캐시·KV 양자화·sink
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-03-kv-cache-design/

python example.py          # 계산기 + 미니 Llama 실측 (다운로드 없음)
python example.py --real   # SmolLM2-135M의 실제 K/V로 KV 양자화, sliding window vs sink 실험
"""
import sys, math, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.util import setup, save, alice_text, load_smollm
from common.roofline import roofline_calc, measure_bw, EDGE_NPU, LLAMA31_8B
from common.tiny_llama import Config, TinyLlama, make_caches, greedy, n_params

GiB = 2 ** 30


def kv_bytes(cfg, ctx, bytes_per=2.0):
    """토큰당 KV = 2(K,V) × 층 × KV헤드 × head_dim × 바이트."""
    return 2 * cfg["n_layers"] * cfg["n_kv_heads"] * cfg["head_dim"] * bytes_per * ctx


def calculators(R):
    c = LLAMA31_8B
    per_tok = kv_bytes(c, 1)
    print(f"[1] Llama-3.1-8B KV(FP16): 토큰당 {per_tok/1024:.0f} KiB, 8K {kv_bytes(c, 8192)/GiB:.2f} GiB, "
          f"128K {kv_bytes(c, 131072)/GiB:.1f} GiB, MHA였다면 128K {kv_bytes(dict(c, n_kv_heads=32), 131072)/GiB:.0f} GiB")
    R["per_tok_KiB"] = per_tok / 1024
    # 용량 예산 — 가정: 가중치 W4 g128 4.14 GB, OS·런타임·활성값 여유 4 GB
    w = c["params"] * 4.125 / 8; reserve = 4e9
    cap = {}
    for mem in (12e9, 16e9, 24e9):
        cap[int(mem / 1e9)] = {name: int((mem - w - reserve) / kv_bytes(c, 1, b)) for name, b in
                              [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]}
        print(f"  메모리 {mem/1e9:.0f} GB → 최대 토큰 FP16 {cap[int(mem/1e9)]['FP16']:,} / INT8 {cap[int(mem/1e9)]['INT8']:,} / INT4 {cap[int(mem/1e9)]['INT4']:,}")
    R["capacity"] = cap
    # 대역폭 예산 — decode마다 KV 전체를 읽는다
    bw = {}
    for ctx in (0, 1024, 4096, 8192, 16384, 32768, 65536, 131072):
        bw[ctx] = {name: roofline_calc(c["params"], 4.125, EDGE_NPU["bw"], EDGE_NPU["tops"],
                                       kv_bytes=kv_bytes(c, ctx, b))["tok_s"]
                   for name, b in [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]}
    print("  대역폭 예산(가상 엣지 NPU 153.6 GB/s, W4 g128): " +
          ", ".join(f"{k}→{v['FP16']:.1f}" for k, v in bw.items()) + " tok/s (FP16 KV)")
    R["bw_budget"] = bw


def tiny_measure(R):
    print("\n[2] 미니 Llama(MHA, KV가 큰 설정)로 컨텍스트별 decode 시간 — 예측 vs 실측")
    cfg = Config(vocab=8192, dim=512, n_layers=8, n_heads=8, n_kv_heads=8, ffn=1408, max_seq=4096)
    model = TinyLlama(cfg).eval()
    P = n_params(cfg); bw = measure_bw()
    kvc = dict(n_layers=cfg.n_layers, n_kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim)
    print(f"  파라미터 {P/1e6:.1f}M (FP32 {P*4/1e6:.0f} MB), 토큰당 KV(FP32) {kv_bytes(kvc,1,4)/1024:.0f} KiB, 측정 BW {bw/1e9:.1f} GB/s")
    rows = []
    # 읽는 KV 바이트 모델: cat = 옛 캐시 읽기+새로 쓰기(복사) + attention 읽기 = 3×KV(ctx)
    #                    정적(전체) = MAXLEN 전체 1×, 정적(slice) = ctx만 1×
    factor = {"dynamic": 3, "static": 1, "sliced": 1}
    with torch.no_grad():
        for ctx in (256, 1024, 2048, 3072, 4000):
            line = f"  ctx {ctx:5d}:"
            for kind in ("dynamic", "static", "sliced"):
                caches = make_caches(cfg, kind)
                model(torch.randint(0, cfg.vocab, (1, ctx)), 0, caches)
                ts = []
                for i in range(min(12, cfg.max_seq - ctx - 1)):
                    t0 = time.perf_counter(); model(torch.tensor([[3]]), ctx + i, caches)
                    ts.append(time.perf_counter() - t0)
                t = sorted(ts[2:])[len(ts[2:]) // 2]
                read_ctx = cfg.max_seq if kind == "static" else ctx
                pred = (P * 4 + factor[kind] * kv_bytes(kvc, read_ctx, 4)) / bw
                rows.append(dict(ctx=ctx, kind=kind, t_ms=t * 1e3, pred_ms=pred * 1e3))
                line += f"  {kind} {t*1e3:6.2f}ms(예측 {pred*1e3:5.2f})"
            print(line)
    # 기울기(1K 토큰당 추가 ms): 고정 오버헤드를 빼고 비교
    for kind in ("dynamic", "sliced"):
        r = [x for x in rows if x["kind"] == kind]
        meas = (r[-1]["t_ms"] - r[0]["t_ms"]) / (r[-1]["ctx"] - r[0]["ctx"]) * 1024
        pred = (r[-1]["pred_ms"] - r[0]["pred_ms"]) / (r[-1]["ctx"] - r[0]["ctx"]) * 1024
        R.setdefault("slope", {})[kind] = dict(meas=meas, pred=pred)
        print(f"  {kind}: 컨텍스트 1K 토큰당 추가 시간 실측 {meas:.2f} ms / 예측 {pred:.2f} ms")
    R["tiny_ctx"] = rows; R["tiny_bw"] = bw
    # 정적 vs 동적: 같은 답인가
    cfg2 = Config(max_seq=256); m2 = TinyLlama(cfg2).eval()
    prompt = torch.randint(0, cfg2.vocab, (1, 20))
    a = greedy(m2, prompt, 40, "static"); b = greedy(m2, prompt, 40, "dynamic"); c3 = greedy(m2, prompt, 40, "sliced"); b = b if b == c3 else None
    with torch.no_grad():
        cs, cd = make_caches(cfg2, "static"), make_caches(cfg2, "dynamic")
        diff = (m2(prompt, 0, cs) - m2(prompt, 0, cd)).abs().max().item()
    print(f"  정적 vs 동적 캐시: 생성 40토큰 일치 = {a == b}, prefill logits 최대 차이 {diff:.1e}")
    R["static_vs_dynamic"] = dict(same=a == b, maxdiff=diff)


# ---------- KV 양자화
def fq(x, bits, dim):
    """dim 축을 따라 absmax 대칭 양자화. K/V shape (H, T, d): dim=-1 → 토큰별, dim=-2 → 채널별."""
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(dim, keepdim=True).clamp(min=1e-8) / qmax
    return (x / s).round().clamp(-qmax - 1, qmax) * s


def attn(q, k, v):
    return ((q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5).softmax(-1) @ v


def kv_quant_table(q, k, v, label):
    ref = attn(q, k, v); res = {}
    for bits in (8, 4, 2):
        for kd, kn in ((-1, "토큰별"), (-2, "채널별")):
            for vd, vn in ((-1, "토큰별"), (-2, "채널별")):
                y = attn(q, fq(k, bits, kd), fq(v, bits, vd))
                res[f"INT{bits} K{kn}/V{vn}"] = ((y - ref).norm() / ref.norm()).item()
    print(f"\n[3] KV 양자화 attention 출력 상대오차 ({label})")
    for kk, vv in res.items():
        print(f"  {kk:22s} {vv:.4f}")
    return res


def synth_kv():
    g = torch.Generator().manual_seed(0)
    H, T, d = 4, 1024, 64
    k = torch.randn(H, T, d, generator=g); k[..., [3, 17, 40]] *= 15    # K에는 채널 아웃라이어(관찰 사례)
    v = torch.randn(H, T, d, generator=g)
    q = torch.randn(H, 16, d, generator=g)
    return q, k, v


def real(R):
    tok, model = load_smollm()
    ids = tok(alice_text(), return_tensors="pt").input_ids[0]
    # (a) 실제 K/V 꺼내서 양자화
    seq = ids[:1024][None]
    with torch.no_grad():
        out = model(seq, use_cache=True)
    kv = out.past_key_values
    layer = 10
    k = kv.layers[layer].keys[0]; v = kv.layers[layer].values[0]        # (Hkv, T, d)
    q = k[:, -16:]                                                        # 근사: 최근 키를 쿼리로
    ch_max = k.abs().amax((0, 1))
    print(f"\n  SmolLM2 layer {layer} K 채널별 max: 중앙값 {ch_max.median():.2f}, 최대 {ch_max.max():.2f}")
    R["real_k_channel_max"] = ch_max.tolist()
    R["kvq_real"] = kv_quant_table(q, k, v, f"SmolLM2-135M layer {layer} 실제 K/V")
    # (b) sliding window vs sink + window : 4D 마스크로 같은 위치(RoPE)는 유지
    T, Wn, sink = 2048, 256, 4
    seq = ids[4096:4096 + T][None]
    def mask(kind):
        i = torch.arange(T)[:, None]; j = torch.arange(T)[None]
        ok = j <= i
        if kind != "full":
            win = (i - j) < Wn
            ok = ok & (win | (j < sink)) if kind == "sink" else ok & win
        return torch.where(ok, 0.0, torch.finfo(torch.float32).min)[None, None]
    res = {}
    with torch.no_grad():
        full_logits = None
        for kind in ("full", "window", "sink"):
            o = model(seq, attention_mask=mask(kind), labels=seq)
            lg = o.logits[0, Wn + 64:]
            # 위치별 손실 (윈도우가 꽉 찬 뒤 구간만)
            lp = torch.log_softmax(o.logits[0, :-1], -1).gather(1, seq[0, 1:, None])[:, 0]
            res[kind] = dict(ppl_all=math.exp(-lp.mean().item()), ppl_after=math.exp(-lp[Wn + 64:].mean().item()),
                             curve=[(-lp[s:s + 128]).mean().item() for s in range(0, T - 1, 128)])
            if kind == "full": full_logits = lg
            res[kind]["logit_diff"] = (lg - full_logits).abs().max().item()
            print(f"  {kind:6s}: PPL(전체) {res[kind]['ppl_all']:.2f}, PPL(위치 {Wn+64}+) {res[kind]['ppl_after']:.2f}, "
                  f"full 대비 logits 최대차 {res[kind]['logit_diff']:.2f}")
    # 첫 토큰이 받는 attention 비중 (sink 관찰)
    _, eager = load_smollm(attn="eager")                  # attention 확률을 꺼내려면 eager 구현
    with torch.no_grad():
        o = eager(ids[4096:4096 + 512][None], output_attentions=True)
    a0 = [o.attentions[l][0, :, 256:, 0].mean().item() for l in range(len(o.attentions))]
    print(f"  위치 0 토큰이 받는 평균 attention(층별): 최소 {min(a0):.2f}, 최대 {max(a0):.2f}, 평균 {sum(a0)/len(a0):.2f}  (균등이면 ~{1/384:.4f})")
    R["sink"] = res; R["attn_to_first"] = a0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); a = ap.parse_args()
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    calculators(R)
    tiny_measure(R)
    R["kvq_synth"] = kv_quant_table(*synth_kv(), "합성 K: 채널 3개 ×15")
    if a.real:
        torch.set_num_threads(8); real(R)
    save(Path(__file__).parent, "03" + ("_real" if a.real else ""), R)
