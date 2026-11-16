"""8장 예제 — 한 번에 여러 토큰 맞히기: 추측 디코딩 (거절 샘플링·기대 토큰 수·실측)
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-08-speculative-decoding/

python example.py          # 장난감 분포 + 시뮬레이터 (다운로드 없음, 수 초)
python example.py --real   # SmolLM2-135M: 층 건너뛰기 draft, prompt lookup draft로 실제 추측 디코딩
"""
import sys, time, copy, argparse, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.util import setup, save, alice_text, load_smollm


def expected_tokens(a, g):
    """한 번 검증에 얻는 기대 토큰 수 E = (1-α^(γ+1)) / (1-α)  (Leviathan et al. 2023)."""
    return (1 - a ** (g + 1)) / (1 - a) if a < 1 else g + 1


def speedup(a, g, c, v=1.0):
    """속도 향상 = E / (γ·c + v). c = draft 1스텝 / target 1스텝, v = γ+1토큰 검증 / 1토큰 (메모리 병목이면 v≈1)."""
    return expected_tokens(a, g) / (g * c + v)


def toy(R):
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(10, generator=g) * 1.5, 0)      # target
    q = torch.softmax(torch.randn(10, generator=g) * 1.5, 0)      # draft
    N = 100000; out = torch.empty(N, dtype=torch.long); acc = 0
    xs = torch.multinomial(q, N, replacement=True, generator=g); u = torch.rand(N, generator=g)
    resid = (p - q).clamp(min=0); resid /= resid.sum()
    rs = torch.multinomial(resid, N, replacement=True, generator=g)
    for i in range(N):
        x = xs[i]
        if u[i] < min(1.0, (p[x] / q[x]).item()):
            out[i] = x; acc += 1
        else:
            out[i] = rs[i]                                        # (p - q)+ 에서 다시 뽑기
    emp = torch.bincount(out, minlength=10).float() / N
    kl = lambda a, b: (a * (a / b).log()).sum().item()
    alpha_pred = torch.minimum(p, q).sum().item()                 # α = Σ min(p, q)
    print(f"[1] 장난감 분포 10만 샘플: KL(p‖경험분포) {kl(p, emp):.2e}, 그냥 q를 썼다면 KL(p‖q) {kl(p, q):.3f}")
    print(f"    수락률 실측 {acc/N:.4f} vs 예측 Σmin(p,q) {alpha_pred:.4f}")
    R["toy"] = dict(p=p.tolist(), q=q.tolist(), emp=emp.tolist(), kl=kl(p, emp), kl_q=kl(p, q),
                    acc=acc / N, alpha_pred=alpha_pred)
    # 기대 토큰 수: 식 vs 몬테카를로
    random.seed(0); mc = {}
    for a in (0.5, 0.7, 0.9):
        for gm in (1, 2, 4, 8):
            tot = 0
            for _ in range(20000):
                k = 0
                while k < gm and random.random() < a: k += 1
                tot += k + 1                                      # 수락 k개 + target이 주는 1개
            mc[f"{a}/{gm}"] = dict(formula=expected_tokens(a, gm), mc=tot / 20000)
    R["mc"] = mc
    print("[2] E 식 vs 몬테카를로:", {k: (round(v["formula"], 3), round(v["mc"], 3)) for k, v in mc.items() if k.endswith("/4")})
    print(f"    예: α=0.7, γ=4, c=0.1 → E={expected_tokens(0.7, 4):.2f}, 속도 {speedup(0.7, 4, 0.1):.2f}배;"
          f" 퀴즈 α=0.8, γ=3, c=0.05 → {speedup(0.8, 3, 0.05):.2f}배")
    R["grid"] = {f"{a:.2f}": {gm: speedup(a, gm, 0.1) for gm in range(1, 11)} for a in (0.5, 0.6, 0.7, 0.8, 0.9)}
    R["best_gamma"] = {f"{a:.1f}/{c}": max(range(1, 16), key=lambda gm: speedup(a, gm, c))
                       for a in (0.6, 0.7, 0.8, 0.9) for c in (0.05, 0.1, 0.2, 0.4)}


# ---------- 실제 모델
def crop_to(cache, L):
    """KV 되감기: 캐시 길이를 L로 줄인다 (transformers 5의 crop은 '지울 개수'를 음수로 받음)."""
    n = cache.get_seq_length()
    if n > L:
        cache.crop(-(n - L))


def build_draft(model, keep):
    d = copy.deepcopy(model)
    d.model.layers = torch.nn.ModuleList([d.model.layers[i] for i in keep])
    d.config.num_hidden_layers = len(keep)
    for i, layer in enumerate(d.model.layers):
        layer.self_attn.layer_idx = i
    return d.eval()


@torch.no_grad()
def greedy_plain(model, ids, n):
    out = model(ids, use_cache=True); past = out.past_key_values; toks = []
    t = out.logits[:, -1:].argmax(-1)
    for _ in range(n):
        toks.append(int(t)); out = model(t, past_key_values=past, use_cache=True); t = out.logits[:, -1:].argmax(-1)
    return toks


@torch.no_grad()
def speculative(target, ids, n, gamma, draft=None, lookup=False, ngram=3):
    """greedy 추측 디코딩: draft가 γ개 제안 → target 1회로 γ+1 위치 검증 → 맞은 데까지 채택, KV 되감기."""
    tout = target(ids[:, :-1], use_cache=True); tpast = tout.past_key_values
    seq = ids[0].tolist(); n0 = len(seq); calls = 0; accepted = 0; proposed = 0; t_draft = 0.0; t_target = 0.0; lens = []
    if draft is not None:
        dpast = draft(ids[:, :-1], use_cache=True).past_key_values
    while len(seq) - n0 < n:
        t0 = time.perf_counter(); prop = []
        if lookup:                                               # prompt lookup: 마지막 n-gram이 앞에 나온 곳 뒤를 복사
            key = seq[-ngram:]
            for s in range(len(seq) - ngram - 1, -1, -1):
                if seq[s:s + ngram] == key:
                    prop = seq[s + ngram:s + ngram + gamma]; break
        else:
            cur = torch.tensor([[seq[-1]]])
            for _ in range(gamma):
                o = draft(cur, past_key_values=dpast, use_cache=True)
                cur = o.logits[:, -1:].argmax(-1); prop.append(int(cur))
        t_draft += time.perf_counter() - t0
        t0 = time.perf_counter()
        inp = torch.tensor([[seq[-1]] + prop]); lens.append(inp.shape[1])
        o = target(inp, past_key_values=tpast, use_cache=True); calls += 1
        t_target += time.perf_counter() - t0
        pred = o.logits[0].argmax(-1).tolist()                  # 위치마다 target의 정답
        k = 0
        while k < len(prop) and prop[k] == pred[k]: k += 1
        new = prop[:k] + [pred[k]]
        accepted += k; proposed += len(prop)
        L = len(seq) - 1 + 1 + k                                 # target 캐시: 채택된 토큰까지만 남김
        seq += new
        crop_to(tpast, L)
        if draft is not None:
            crop_to(dpast, L - 1 if k == len(prop) else L)
            if k == len(prop):                                   # draft가 마지막 제안을 캐시에 안 넣었으니 보충
                draft(torch.tensor([[prop[-1]]]), past_key_values=dpast, use_cache=True)
    return seq[n0:n0 + n], dict(calls=calls, acc_rate=accepted / max(proposed, 1),
                                tok_per_call=(len(seq) - n0) / calls, t_draft=t_draft, t_target=t_target, verify_lens=lens)


def real(R):
    tok, model = load_smollm(); torch.set_num_threads(1)
    text = alice_text(); ids = tok(text, return_tensors="pt").input_ids[:, 2000:2128]
    N = 64
    # 검증 비용: 토큰 T개를 한 번에 넣을 때 target 1회 시간
    past = model(ids, use_cache=True).past_key_values; L0 = ids.shape[1]; cost = {}
    for T in (1, 2, 3, 5, 9, 12):
        ts = []
        for _ in range(6):
            t0 = time.perf_counter(); model(ids[:, :T], past_key_values=past, use_cache=True); ts.append(time.perf_counter() - t0)
            crop_to(past, L0)
        cost[T] = sorted(ts)[3]
    print("\n[3] target 1회 시간(토큰 T개 검증):", {k: f"{v*1e3:.1f}ms" for k, v in cost.items()})
    R["verify_cost"] = {k: v * 1e3 for k, v in cost.items()}
    def timed(fn, reps=3):
        ts, out = [], None
        for _ in range(reps):
            t0 = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - t0)
        return out, sorted(ts)[reps // 2]
    import numpy as np
    Ts = sorted(cost); interp = lambda T: float(np.interp(T, Ts, [cost[x] for x in Ts]))
    def account(st):          # 예측 시간 = draft 시간 + Σ 검증 비용(T_i) — 측정한 비용표로 계산
        return st["t_draft"] + sum(interp(T) for T in st["verify_lens"])
    ref, t_plain = timed(lambda: greedy_plain(model, ids, N))
    print(f"[4] 일반 greedy {N}토큰: {t_plain*1e3:.0f} ms ({N/t_plain:.1f} tok/s)")
    res = {"plain": dict(ms=t_plain * 1e3, tps=N / t_plain)}
    nL = model.config.num_hidden_layers
    cases = [("[실패] 층 건너뛰기 draft 6층", list(range(3)) + list(range(nL - 3, nL)), None),
             ("[실패] 층 건너뛰기 draft 16층", list(range(0, nL, 2)) + [nL - 1], None),
             ("[실패] 층 건너뛰기 draft 26층", [i for i in range(nL) if i not in (10, 14, 18, 22)], None),
             ("[실패] BF16 복제 draft(30층)", None, torch.bfloat16)]
    gm = 4; v = cost[gm + 1] / cost[1]
    R["v"] = v
    print(f"  검증 비용 v(γ=4) = target(5토큰)/target(1토큰) = {v:.2f}  (가상 엣지 NPU처럼 ridge가 크면 ≈1)")
    for name, keep, dt in cases:
        d = build_draft(model, keep) if keep else copy.deepcopy(model).to(dt).eval()
        (out, st), t = timed(lambda: speculative(model, ids, N, gm, draft=d), reps=2)
        ct = (st["t_draft"] / (st["calls"] * gm)) / cost[1]
        pred = speedup(st["acc_rate"], gm, ct, v)
        res[name] = dict(layers=len(keep) if keep else nL, gamma=gm, same=out == ref, ms=t * 1e3, speed=t_plain / t,
                         c=ct, pred=pred, pred_ideal=speedup(st["acc_rate"], gm, ct, 1.0),
                         pred_account=N * cost[1] / account(st), **{k: v for k, v in st.items() if k != "verify_lens"})
        print(f"  {name:24s} 일치 {out == ref}, 수락률 {st['acc_rate']:.2f}, 호출당 {st['tok_per_call']:.2f}토큰, "
              f"c≈{ct:.2f}, 예측(식) {pred:.2f}배 / 예측(비용표) {N*cost[1]/account(st):.2f}배 / 실측 {t_plain/t:.2f}배")
        del d
    # prompt lookup: 입력을 그대로 다시 쓰는 과제
    passage = tok.decode(ids[0])
    prompt = passage + "\n\nHere is the same passage, copied exactly:\n\n" + passage[:40]
    pid = tok(prompt, return_tensors="pt").input_ids
    ref2, tp2 = timed(lambda: greedy_plain(model, pid, N))
    for gm in (4, 8):
        (out, st), t = timed(lambda: speculative(model, pid, N, gm, lookup=True), reps=2)
        pa = N * cost[1] / account(st)
        res[f"prompt lookup γ={gm}"] = dict(same=out == ref2, ms=t * 1e3, speed=tp2 / t, plain_ms=tp2 * 1e3,
                                            pred_account=pa, **{k: v for k, v in st.items() if k != "verify_lens"})
        print(f"  prompt lookup γ={gm} (복사 과제): 일치 {out == ref2}, 수락률 {st['acc_rate']:.2f}, 호출당 {st['tok_per_call']:.2f}토큰, 예측(비용표) {pa:.2f}배 / 실측 {tp2/t:.2f}배")
    (out, st), t = timed(lambda: speculative(model, ids, N, 4, lookup=True), reps=2)
    pa = N * cost[1] / account(st)
    res["prompt lookup γ=4 (일반 이어쓰기)"] = dict(same=out == ref, ms=t * 1e3, speed=t_plain / t, pred_account=pa,
                                                **{k: v for k, v in st.items() if k != "verify_lens"})
    print(f"  prompt lookup γ=4 (일반 이어쓰기): 일치 {out == ref}, 수락률 {st['acc_rate']:.2f}, 호출당 {st['tok_per_call']:.2f}토큰, 예측(비용표) {pa:.2f}배 / 실측 {t_plain/t:.2f}배")
    R["real"] = res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); a = ap.parse_args()
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    toy(R)
    if a.real:
        real(R)
    save(Path(__file__).parent, "08" + ("_real" if a.real else ""), R)
