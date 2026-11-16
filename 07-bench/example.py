"""7장 예제 — 숫자를 정직하게 재는 법: TTFT·TPOT·decode TPS·E2E TPS 측정 하네스
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-07-ttft-tps-benchmark/

python example.py          # 미니 Llama (다운로드 없음)
python example.py --real   # SmolLM2-135M (+ 토크나이저 비교용 Qwen2.5 토크나이저 파일만 받음)
python example.py --llama-bench ../tools/bench.json   # llama-bench -o json 결과를 같은 표로
"""
import sys, time, json, argparse, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.util import setup, save, alice_text, load_smollm
from common.roofline import measure_bw, measure_peak, roofline_calc
from common.tiny_llama import Config, TinyLlama, make_caches, n_params


# ---------- 하네스: 어떤 generate_stream()에도 붙는 타임스탬프 기록기
class Recorder:
    def __init__(self, n_prompt):
        self.n_prompt = n_prompt; self.t0 = None; self.ts = []

    def start(self): self.t0 = time.perf_counter()
    def on_token(self, _tok): self.ts.append(time.perf_counter())

    def metrics(self):
        N = len(self.ts); e2e = self.ts[-1] - self.t0; ttft = self.ts[0] - self.t0
        tpot = (e2e - ttft) / (N - 1) if N > 1 else float("nan")
        itl = [b - a for a, b in zip(self.ts, self.ts[1:])]
        return dict(n_prompt=self.n_prompt, n_out=N, ttft_s=ttft, tpot_s=tpot, decode_tps=1 / tpot,
                    e2e_s=e2e, e2e_tps=N / e2e, total_tps=(self.n_prompt + N) / e2e,
                    itl_p50=st.median(itl) if itl else None, itl_p90=sorted(itl)[int(0.9 * len(itl))] if itl else None)


@torch.no_grad()
def stream_tiny(model, prompt, n_new, rec):
    caches = make_caches(model.cfg, "sliced"); rec.start()
    lo = model(prompt, 0, caches); tok = int(lo[0, -1].argmax()); rec.on_token(tok); pos = prompt.shape[1]
    for _ in range(n_new - 1):
        lo = model(torch.tensor([[tok]]), pos, caches); pos += 1; tok = int(lo[0, -1].argmax()); rec.on_token(tok)


@torch.no_grad()
def stream_hf(model, prompt, n_new, rec):
    rec.start()
    out = model(prompt, use_cache=True); past = out.past_key_values
    tok = out.logits[:, -1:].argmax(-1); rec.on_token(int(tok))
    for _ in range(n_new - 1):
        out = model(tok, past_key_values=past, use_cache=True); past = out.past_key_values
        tok = out.logits[:, -1:].argmax(-1); rec.on_token(int(tok))


def run(stream, model, prompt, n_new):
    rec = Recorder(prompt.shape[1]); stream(model, prompt, n_new, rec); return rec.metrics()


def suite(R, stream, model, make_prompt, w_bytes, flops_per_tok, label):
    bw = measure_bw(); peak = measure_peak()
    print(f"\n=== {label}: 측정 BW {bw/1e9:.1f} GB/s, matmul 피크 {peak/1e9:.0f} GFLOP/s")
    # 1) 한 번의 실행 로그 → 정의 3가지
    run(stream, model, make_prompt(128), 8)                     # 웜업
    m = run(stream, model, make_prompt(128), 64)
    print(f"[1] 프롬프트 128, 출력 64: TTFT {m['ttft_s']*1e3:.1f} ms, TPOT {m['tpot_s']*1e3:.2f} ms")
    print(f"    decode TPS {m['decode_tps']:.1f} / E2E TPS {m['e2e_tps']:.1f} / (입력+출력)/E2E {m['total_tps']:.1f}")
    out = {"one_run": m}
    # 2) 반복 통계: 첫 실행(콜드) 포함/제외
    reps = []
    for i in range(8):
        reps.append(run(stream, model, make_prompt(128), 32)["decode_tps"])
    out["reps"] = reps
    print(f"[2] 8회 반복 decode TPS: 중앙값 {st.median(reps):.1f}, 최소 {min(reps):.1f}, 최대 {max(reps):.1f}, p90 {sorted(reps)[int(0.9*8)]:.1f}")
    # 3) 프롬프트 길이 스윕 → TTFT, 예측(2·P·N/피크)
    sweep = []
    for P in (32, 64, 128, 256, 512, 1024):
        mm = run(stream, model, make_prompt(P), 2)
        pred = flops_per_tok * P / peak
        sweep.append(dict(P=P, ttft_ms=mm["ttft_s"] * 1e3, pred_ms=pred * 1e3))
        print(f"[3] P={P:5d} TTFT 실측 {mm['ttft_s']*1e3:7.1f} ms / 계산 하한 {pred*1e3:7.1f} ms")
    out["ttft_sweep"] = sweep
    # 4) 컨텍스트 위치 스윕 → decode TPS
    ctxs = []
    for P in (64, 512, 1024, 1536):
        mm = run(stream, model, make_prompt(P), 16)
        ctxs.append(dict(ctx=P, decode_tps=mm["decode_tps"]))
        print(f"[4] 컨텍스트 {P:5d}에서 decode TPS {mm['decode_tps']:.1f}")
    out["ctx_sweep"] = ctxs
    # 5) 예측 vs 실측 (Roofline: 가중치 바이트 / 측정 BW)
    pred_tps = bw / w_bytes
    out["roofline"] = dict(pred_tps=pred_tps, meas_tps=st.median(reps), util=st.median(reps) / pred_tps, bw=bw, peak=peak)
    print(f"[5] decode 상한 예측 {pred_tps:.1f} tok/s vs 실측 중앙값 {st.median(reps):.1f} → 대역폭 이용률 {st.median(reps)/pred_tps:.0%}")
    # 실패 장면: TTFT 포함 평균 TPS로 보고하면 답이 짧을수록 낮게 나옴
    fail = []
    for n in (4, 16, 64, 256):
        mm = run(stream, model, make_prompt(512), n)
        fail.append(dict(n_out=n, e2e_tps=mm["e2e_tps"], decode_tps=mm["decode_tps"], ttft_ms=mm["ttft_s"] * 1e3))
        print(f"[실패 장면] 프롬프트 512, 출력 {n:4d}: E2E TPS {mm['e2e_tps']:6.1f} vs decode TPS {mm['decode_tps']:6.1f}")
    out["fail"] = fail
    R[label] = out


def tokenizer_compare(R):
    from transformers import AutoTokenizer
    toks = {"SmolLM2-135M": AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M"),
            "Qwen2.5-0.5B": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")}
    texts = {"영어(앨리스 1문단)": alice_text()[200:1200],
             "한국어": "온디바이스 LLM은 작은 기기 안에서 언어 모델을 직접 돌리는 기술이다. 메모리 대역폭이 속도를 정하고, "
                     "양자화와 KV 캐시 설계가 품질과 길이를 정한다. 같은 문장도 토크나이저가 다르면 토큰 수가 달라진다." * 3}
    res = {}
    for tn, tk in toks.items():
        for xn, x in texts.items():
            n = len(tk(x).input_ids); res[f"{tn} / {xn}"] = dict(tokens=n, chars=len(x), chars_per_tok=len(x) / n)
            print(f"[6] {tn:13s} {xn:12s}: 글자 {len(x):5d}, 토큰 {n:5d}, 토큰당 글자 {len(x)/n:.2f}")
    R["tokenizers"] = res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); ap.add_argument("--llama-bench")
    a = ap.parse_args()
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    cfg = Config(vocab=32000, dim=1024, n_layers=8, n_heads=16, n_kv_heads=4, ffn=2816, max_seq=2048)
    tiny = TinyLlama(cfg).eval(); P = n_params(cfg)
    g = torch.Generator().manual_seed(0)
    suite(R, stream_tiny, tiny, lambda n: torch.randint(0, cfg.vocab, (1, n), generator=g), P * 4, 2 * P, "tiny")
    if a.real:
        tok, model = load_smollm()
        ids = tok(alice_text(), return_tensors="pt").input_ids
        Pr = sum(p.numel() for p in model.parameters())
        suite(R, stream_hf, model, lambda n: ids[:, 1000:1000 + n], Pr * 4, 2 * Pr, "smollm2")
        tokenizer_compare(R)
    if a.llama_bench:
        d = json.load(open(a.llama_bench)); rows = []
        for r in d:
            rows.append(dict(model=Path(r["model_filename"]).name, threads=r["n_threads"],
                             test=f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}",
                             tps=r["avg_ts"], sd=r["stddev_ts"], size_MB=r["model_size"] / 1e6))
            print(f"[llama-bench] {rows[-1]['model']:28s} t={rows[-1]['threads']} {rows[-1]['test']:6s} {r['avg_ts']:8.1f} ± {r['stddev_ts']:.1f} t/s")
        R["llama_bench"] = rows
    save(Path(__file__).parent, "07" + ("_real" if a.real else ""), R)
