"""공용 Roofline 계산기 — 1장에서 만들고 3·7·8·9장이 다시 씀.

모든 본문 숫자는 이 함수 출력에서 가져온다(손계산 금지).
"""
import time
import torch

# 가상 엣지 NPU (공개 스펙 수준의 숫자를 쓴 '예시' 장치)
# 가정: 80 TOPS는 INT8 기준(정밀도는 공개 자료에 없음), LPDDR5X-9600 × 128bit(8×16bit 채널)
EDGE_NPU = dict(name="가상 엣지 NPU", tops=80e12, bw=153.6e9, power_w=5.0)
EDGE_NPU_HALF = dict(name="가상 엣지 NPU (4채널)", tops=80e12, bw=76.8e9, power_w=5.0)

LLAMA31_8B = dict(name="Llama-3.1-8B", params=8.03e9, n_layers=32, hidden=4096,
                  ffn=14336, n_heads=32, n_kv_heads=8, head_dim=128, vocab=128256)


def lpddr_bw(mts=9600, bus_bits=128):
    """대역폭 = 전송률(MT/s) × 버스 폭(byte)."""
    return mts * 1e6 * bus_bits / 8


def roofline_calc(params, bits, bw, tops, kv_bytes=0.0, util_bw=1.0, util_tops=1.0):
    """decode 1토큰 = 가중치 전부 + KV 전부를 한 번 읽는다고 보고 상한을 계산."""
    w_bytes = params * bits / 8
    bytes_per_tok = w_bytes + kv_bytes
    tok_s_mem = bw * util_bw / bytes_per_tok
    flops_per_tok = 2 * params
    tok_s_cmp = tops * util_tops / flops_per_tok
    ridge = tops / bw                                  # FLOP/byte
    ai_decode = flops_per_tok / bytes_per_tok
    return dict(weight_GB=w_bytes / 1e9, bytes_per_tok_GB=bytes_per_tok / 1e9,
                tok_s=min(tok_s_mem, tok_s_cmp), tok_s_mem=tok_s_mem, tok_s_cmp=tok_s_cmp,
                need_TOPS=flops_per_tok * min(tok_s_mem, tok_s_cmp) / 1e12,
                ridge=ridge, ai_decode=ai_decode,
                prefill_cross_N=ridge / ai_decode)


def ttft_lower(params, n_prompt, tops, util=1.0):
    return 2 * params * n_prompt / (tops * util)


# ---------- 이 PC 재기
def timeit(fn, reps=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


def measure_bw(mb=256):
    """큰 텐서 복사 = 읽기 + 쓰기. 바이트/시간 = 실효 대역폭."""
    a = torch.empty(mb * 2**20 // 4); b = torch.empty_like(a)
    t = timeit(lambda: b.copy_(a), reps=7)
    return 2 * a.numel() * 4 / t


def measure_peak(n=2048):
    a = torch.randn(n, n); b = torch.randn(n, n)
    t = timeit(lambda: a @ b, reps=5)
    return 2 * n ** 3 / t
