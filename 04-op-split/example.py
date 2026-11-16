"""4장 예제 — NPU가 못 하는 몇 %가 시간을 먹는다: 연산 분류·Amdahl·경계 비용·비선형 근사
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-04-npu-cpu-dsp-split/

python example.py   (CPU, 다운로드 없음, ~30초)
"""
import sys, time, threading, queue
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity
from common.util import setup, save
from common.roofline import LLAMA31_8B
from common.tiny_llama import Config, TinyLlama, make_caches

CAT = {"matmul": ("mm", "bmm", "matmul", "linear", "addmm"),
       "축소(reduction)": ("mean", "amax", "sum", "softmax", "_softmax", "rsqrt", "pow"),
       "원소별": ("mul", "add", "sub", "silu", "neg", "div", "where", "exp", "cos", "sin"),
       "메모리·모양": ("copy_", "slice", "select", "stack", "cat", "view", "reshape", "transpose",
                    "index", "embedding", "flatten", "unsqueeze", "clone", "contiguous", "arange",
                    "expand", "_unsafe_view", "zeros", "empty", "as_strided", "t", "lift_fresh", "detach")}


def categorize(name):
    n = name.replace("aten::", "")
    for c, keys in CAT.items():
        if n in keys or n.rstrip("_") in keys:
            return c
    return "기타"


def profile_decode(model, threads):
    torch.set_num_threads(threads)
    cfg = model.cfg; caches = make_caches(cfg, "sliced")
    with torch.no_grad():
        model(torch.randint(0, cfg.vocab, (1, 512)), 0, caches)
        for i in range(5): model(torch.tensor([[1]]), 512 + i, caches)
        t0 = time.perf_counter()
        for i in range(20): model(torch.tensor([[1]]), 517 + i, caches)
        wall = (time.perf_counter() - t0) / 20
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for i in range(20): model(torch.tensor([[1]]), 537 + i, caches)
    agg = {}
    for e in prof.key_averages():
        if e.key.startswith("aten::"):
            c = categorize(e.key)
            agg[c] = agg.get(c, 0) + e.self_cpu_time_total / 20 / 1e3     # ms/token
    return wall * 1e3, agg


def analytic_shares(c, ctx=1024):
    """Llama-3.1-8B 1토큰 decode의 연산 종류별 FLOP·바이트(가중치 W4, 활성값 FP16)."""
    L, d, f, H, Hk, hd = c["n_layers"], c["hidden"], c["ffn"], c["n_heads"], c["n_kv_heads"], c["head_dim"]
    fl = {"matmul": 2 * c["params"] + L * 2 * 2 * H * hd * ctx,          # 가중치 GEMV + QK·PV
          "축소(reduction)": L * (2 * 4 * d + 5 * H * ctx),                # RMSNorm 2번 + softmax
          "원소별": L * (6 * (H + Hk) * hd + 5 * f + 2 * d)}               # RoPE + SwiGLU + residual
    by = {"matmul": c["params"] * 4.125 / 8 + L * 2 * Hk * hd * ctx * 2,
          "축소(reduction)": L * (2 * 2 * 2 * d + 2 * 2 * H * ctx),
          "원소별": L * (2 * 2 * (H + Hk) * hd + 3 * 2 * f + 3 * 2 * d)}
    return fl, by


def amdahl(f, s):
    return 1 / ((1 - f) + f / s)


if __name__ == "__main__":
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    # ---------- 1) 프로파일
    cfg = Config(vocab=8192, dim=1024, n_layers=8, n_heads=16, n_kv_heads=4, ffn=2816, max_seq=1024)
    model = TinyLlama(cfg).eval()
    prof = {}
    for th in (1, 8):
        wall, agg = profile_decode(model, th); prof[th] = dict(wall=wall, agg=agg)
        tot = sum(agg.values())
        print(f"\n[1] decode 1토큰 프로파일 ({th}스레드): wall {wall:.2f} ms, op self 합 {tot:.2f} ms")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"  {k:14s} {v:6.2f} ms  ({v/tot:.0%})")
    R["profile"] = prof
    torch.set_num_threads(1)
    fl, by = analytic_shares(LLAMA31_8B)
    print("\n[2] Llama-3.1-8B 1토큰(ctx 1024) 분석치: FLOP·바이트 비중")
    for k in fl:
        print(f"  {k:14s} FLOP {fl[k]/sum(fl.values()):.3%}  바이트 {by[k]/sum(by.values()):.3%}")
    R["analytic"] = dict(flops=fl, bytes=by)
    # ---------- 2) Amdahl: 예측 vs 실측 (1→8스레드에서 matmul만 크게 빨라진다고 보고)
    a1, a8 = prof[1]["agg"], prof[8]["agg"]
    f = a1["matmul"] / sum(a1.values()); s = a1["matmul"] / a8["matmul"]
    pred = amdahl(f, s); meas = sum(a1.values()) / sum(a8.values())
    print(f"\n[3] Amdahl: matmul 비중 f={f:.2f}, matmul 가속 s={s:.2f} → 예측 {pred:.2f}배, 실측(op 합) {meas:.2f}배, 실측(wall) {prof[1]['wall']/prof[8]['wall']:.2f}배")
    R["amdahl_check"] = dict(f=f, s=s, pred=pred, meas=meas, wall=prof[1]["wall"] / prof[8]["wall"])
    R["amdahl_table"] = {ff: {ss: amdahl(ff, ss) for ss in (5, 10, 50, 1e9)} for ff in (0.8, 0.9, 0.95, 0.99)}
    print("  f=0.95, s=50 →", round(amdahl(0.95, 50), 1), "배 / s=∞ →", round(amdahl(0.95, 1e12), 1), "배")
    # ---------- 3) 경계 넘기 비용
    x = torch.randn(4096)
    t_op = min(timeit_ for timeit_ in [(lambda: (lambda t0: [x.add(1.0) for _ in range(10000)] and (time.perf_counter() - t0) / 10000)(time.perf_counter()))() for _ in range(3)])
    q_in, q_out = queue.Queue(), queue.Queue()
    def worker():
        while True:
            item = q_in.get()
            if item is None: break
            q_out.put(item + 1.0)
    th = threading.Thread(target=worker); th.start()
    for _ in range(100): q_in.put(x); q_out.get()
    t0 = time.perf_counter()
    for _ in range(2000): q_in.put(x); q_out.get()
    t_hand = (time.perf_counter() - t0) / 2000
    q_in.put(None); th.join()
    print(f"\n[4] 8KB(4096 FP16 상당) 텐서: op 1회 {t_op*1e6:.1f} µs, 다른 스레드로 넘겼다 받기 왕복 {t_hand*1e6:.1f} µs")
    R["boundary"] = dict(op_us=t_op * 1e6, handoff_us=t_hand * 1e6)
    base = 1 / 37.1                                       # 1장 상한(가상 엣지 NPU)에서의 토큰당 시간
    sim = {}
    for lat_us in (5, 20, 50, 100):
        for n_cross in (2, 4, 8):
            t = base + 32 * n_cross * lat_us * 1e-6
            sim[f"{lat_us}us x {n_cross}/층"] = 1 / t
    R["boundary_sim"] = sim
    print("  가정: 경계 지연 20µs × 층당 4번 × 32층 →", f"{sim['20us x 4/층']:.1f} tok/s (경계 없으면 37.1)")
    # ---------- 4) 비선형 근사 오차: 예측 vs 실측
    xs = torch.linspace(-16, 0, 100001, dtype=torch.float64)
    for n in (64, 256, 1024):
        grid = torch.linspace(-16, 0, n, dtype=torch.float64); tab = grid.exp(); h = 16 / (n - 1)
        idx = ((xs + 16) / h).floor().clamp(max=n - 2).long(); fr = (xs + 16) / h - idx
        approx = tab[idx] * (1 - fr) + tab[idx + 1] * fr
        meas = (approx - xs.exp()).abs().max().item(); pred = h ** 2 / 8      # 선형보간 오차 ≤ h²/8·max|f''|
        R.setdefault("exp_lut", {})[n] = dict(pred=pred, meas=meas)
        print(f"  exp LUT {n:5d}칸 선형보간: 오차 상한 예측 h²/8={pred:.2e}, 실측 최대 {meas:.2e}")
    # online softmax == 2-pass softmax
    s_ = torch.randn(4096, dtype=torch.float64) * 5
    m, l = -float("inf"), 0.0
    for chunk in s_.split(256):                           # Milakov & Gimelshein 2018
        m_new = max(m, chunk.max().item()); l = l * np.exp(m - m_new) + (chunk - m_new).exp().sum().item(); m = m_new
    online = (s_ - m).exp() / l
    print(f"  online softmax(256개씩) vs softmax 최대 차이 {(online - s_.softmax(0)).abs().max().item():.1e}")
    R["online_softmax_diff"] = (online - s_.softmax(0)).abs().max().item()
    # INT8 점수로 softmax
    sc = torch.randn(64, 1024) * 3; ref = sc.softmax(-1)
    for bits in (8, 6, 4):
        qm = 2 ** (bits - 1) - 1; st = sc.abs().amax(-1, keepdim=True) / qm
        e = ((sc / st).round() * st).softmax(-1)
        R.setdefault("softmax_int", {})[bits] = (e - ref).abs().sum(-1).mean().item()
        print(f"  softmax 입력 INT{bits}: 확률분포 L1 오차 평균 {R['softmax_int'][bits]:.4f}")
    # RMSNorm 정수 누산 폭
    d = 4096; xi = torch.randint(-127, 128, (d,), dtype=torch.int32)
    true = int((xi.long() ** 2).sum()); worst = d * 127 ** 2
    acc16 = np.int16(0)
    with np.errstate(over="ignore"):
        for v in xi.numpy().astype(np.int16):
            acc16 = np.int16(acc16 + np.int16(v) * np.int16(v))
    print(f"  RMSNorm 제곱합(d={d}, INT8 입력): 정답 {true:,}, 최악 {worst:,} (INT16 한계 32,767), INT16 누산 결과 {int(acc16):,}")
    R["rms_acc"] = dict(true=true, worst=worst, int16=int(acc16))
    # RoPE 각도를 FP16으로 계산하면
    inv = 1.0 / (500000 ** (torch.arange(0, 128, 2).double() / 128))
    for pos in (100, 1000, 8000, 32000):
        a64 = pos * inv; a16 = (torch.tensor(float(pos), dtype=torch.float16) * inv.half()).double()
        err = (a64 - a16).abs().max().item()
        R.setdefault("rope_fp16", {})[pos] = dict(err=err, pred=pos * 2 ** -11)
        print(f"  RoPE 각도 pos={pos:6d}: FP16 계산 최대 오차 {err:.3f} rad (상한 추정 pos·2⁻¹¹ = {R['rope_fp16'][pos]['pred']:.3f})")
    R["rope_table_MiB"] = 8192 * 64 * 2 * 4 / 2 ** 20
    R["lmhead_KB"] = LLAMA31_8B["vocab"] * 2 / 1024
    print(f"  RoPE 테이블(8K×64쌍, cos+sin FP32) {R['rope_table_MiB']:.0f} MiB, lm_head logits FP16 {R['lmhead_KB']:.0f} KiB/토큰")
    # ---------- 실패 장면: max 빼기 없는 FP16 softmax
    logits = torch.tensor([3.0, 8.0, 11.0, 12.0], dtype=torch.float16)
    naive = logits.exp() / logits.exp().sum()
    safe = (logits - logits.max()).exp() / (logits - logits.max()).exp().sum()
    print(f"\n[실패 장면] FP16 softmax {logits.tolist()}: 그냥 = {naive.tolist()}, max 빼기 = {[round(v, 4) for v in safe.float().tolist()]}")
    print(f"  exp(12) = {np.exp(12):.0f} > FP16 최대 65504 → inf, inf/inf = NaN")
    R["fail_softmax"] = dict(naive=[str(v) for v in naive.tolist()], safe=safe.float().tolist())
    save(Path(__file__).parent, "04", R)
