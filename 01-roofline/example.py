"""1장 예제 — 80 TOPS인데 왜 30 tok/s일까: Roofline과 decode 상한
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-01-roofline-bandwidth/

python example.py   (CPU 1스레드, 다운로드 없음, 1분 이내)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.util import setup, save
from common.roofline import (roofline_calc, ttft_lower, measure_bw, measure_peak, timeit,
                             EDGE_NPU, EDGE_NPU_HALF, LLAMA31_8B, lpddr_bw)
from common.tiny_llama import Config, TinyLlama, make_caches, n_params

info = setup(threads=1)
print("환경:", info)
R = {"env": info}

# ---------- 1) 가상 엣지 NPU에 공개 모델 config 대입
print("\n[1] 가상 엣지 NPU 상한 (가정: 80 TOPS = INT8 피크, 이용률 100%)")
print("LPDDR5X-9600 x 128bit =", lpddr_bw(9600, 128) / 1e9, "GB/s")
rows = []
for name, params, bits, dev in [
        ("Llama-3.1-8B W4 g128", LLAMA31_8B["params"], 4.125, EDGE_NPU),
        ("Llama-3.1-8B W4 g128", LLAMA31_8B["params"], 4.125, EDGE_NPU_HALF),
        ("Llama-3.1-8B W8", LLAMA31_8B["params"], 8.0, EDGE_NPU),
        ("Llama-3.1-8B FP16", LLAMA31_8B["params"], 16.0, EDGE_NPU),
        ("밀집 20B INT4", 20e9, 4.0, EDGE_NPU),
        ("gpt-oss-20b 활성 3.6B, ~4.25bit", 3.6e9, 4.25, EDGE_NPU)]:
    r = roofline_calc(params, bits, dev["bw"], dev["tops"])
    rows.append(dict(model=name, bw=dev["bw"] / 1e9, **{k: round(v, 3) for k, v in r.items()}))
    print(f"  {name:32s} BW {dev['bw']/1e9:6.1f} GB/s  가중치 {r['weight_GB']:6.2f} GB  "
          f"상한 {r['tok_s']:6.1f} tok/s  필요 {r['need_TOPS']:.2f} TOPS  AI {r['ai_decode']:.2f}")
R["npu_table"] = rows
r8 = roofline_calc(LLAMA31_8B["params"], 4.125, EDGE_NPU["bw"], EDGE_NPU["tops"])
print(f"  ridge point = {r8['ridge']:.1f} FLOP/B, prefill 교차점 N ≈ {r8['prefill_cross_N']:.0f} 토큰")
for u in (1.0, 0.4):
    print(f"  TTFT 하한(1000토큰, 이용률 {u:.0%}) = {ttft_lower(LLAMA31_8B['params'], 1000, 80e12, u):.2f} s")
R["ridge"] = r8["ridge"]; R["cross_N"] = r8["prefill_cross_N"]
R["ttft"] = {u: ttft_lower(LLAMA31_8B["params"], 1000, 80e12, u) for u in (1.0, 0.4)}
# 실패 장면: TOPS만 2배
r_2x = roofline_calc(LLAMA31_8B["params"], 4.125, EDGE_NPU["bw"], 2 * EDGE_NPU["tops"])
r_bw = roofline_calc(LLAMA31_8B["params"], 4.125, 2 * EDGE_NPU["bw"], EDGE_NPU["tops"])
print(f"  TOPS 2배 → {r_2x['tok_s']:.1f} tok/s, 대역폭 2배 → {r_bw['tok_s']:.1f} tok/s")
R["fail_2x"] = dict(base=r8["tok_s"], tops2=r_2x["tok_s"], bw2=r_bw["tok_s"])

# ---------- 2) 이 PC의 Roofline
print("\n[2] 이 PC 재기 (CPU 1스레드)")
bw = measure_bw(); peak = measure_peak()
print(f"  복사 대역폭 {bw/1e9:.1f} GB/s, matmul 피크 {peak/1e9:.1f} GFLOP/s, ridge {peak/bw:.1f} FLOP/B")
R["pc"] = dict(bw=bw, peak=peak, ridge=peak / bw)
K, N = 8192, 16384                      # 가중치 512 MB > 이 PC의 L3(320 MiB) → DRAM에서 읽게
W = torch.randn(K, N)
pts = []
for M in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    x = torch.randn(M, K)
    t = timeit(lambda: x @ W, reps=5, warmup=2)
    flops = 2 * M * K * N; byts = 4 * (K * N + M * K + M * N)
    pts.append(dict(M=M, t_ms=t * 1e3, gflops=flops / t / 1e9, ai=flops / byts, gbs=byts / t / 1e9))
    print(f"  M={M:4d}  {t*1e3:7.2f} ms  {flops/t/1e9:7.1f} GFLOP/s  AI {flops/byts:6.1f}  실효 {byts/t/1e9:5.1f} GB/s")
R["gemm_sweep"] = pts

# ---------- 3) 실패 장면 재현: "계산을 더 하면(=TOPS를 더 쓰면) 그만큼 느려진다/빨라진다"는 착각
print("\n[3] 같은 가중치를 한 번 읽으면서 계산만 M배")
t1 = pts[0]["t_ms"]
for p_ in pts[:5]:
    print(f"  계산 {p_['M']:3d}배 → 시간 {p_['t_ms']/t1:.2f}배")
R["fail_compute"] = [dict(M=p_["M"], t_ratio=p_["t_ms"] / t1) for p_ in pts]
del W

# ---------- 4) 예측 vs 실측: tiny decoder의 decode 토큰당 시간
print("\n[4] 미니 Llama decode: 예측(가중치 바이트 / 실측 대역폭) vs 실측")
cfg = Config(vocab=32000, dim=1024, n_layers=8, n_heads=16, n_kv_heads=4, ffn=2816, max_seq=256)
model = TinyLlama(cfg).eval()
P = n_params(cfg); wbytes = P * 4
caches = make_caches(cfg)
with torch.no_grad():
    model(torch.randint(0, cfg.vocab, (1, 16)), 0, caches)
    ts = []
    for i in range(40):
        t0 = time.perf_counter(); model(torch.tensor([[5]]), 16 + i, caches); ts.append(time.perf_counter() - t0)
t_meas = sorted(ts[5:])[len(ts[5:]) // 2]
t_pred = wbytes / bw
print(f"  파라미터 {P/1e6:.1f}M (FP32 {wbytes/1e6:.0f} MB)")
print(f"  예측 {t_pred*1e3:.1f} ms/token ({1/t_pred:.1f} tok/s)  실측 {t_meas*1e3:.1f} ms/token ({1/t_meas:.1f} tok/s)"
      f"  → 대역폭 이용률 {t_pred/t_meas:.0%}")
R["tiny_decode"] = dict(params=P, wbytes=wbytes, t_pred=t_pred, t_meas=t_meas, util=t_pred / t_meas)
save(Path(__file__).parent, "01", R)
