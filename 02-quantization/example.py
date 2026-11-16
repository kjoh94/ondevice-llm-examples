"""2장 예제 — 4비트로 줄이고도 답을 지키는 법: RTN·group·GPTQ·AWQ·SmoothQuant 직접 구현
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-02-llm-quantization/

python example.py          # 합성 Linear (다운로드 없음, ~20초)
python example.py --real   # SmolLM2-135M perplexity 비교 (~270MB 다운로드, 수 분)
모든 양자화는 fake-quant(양자화→즉시 역양자화)로 검증한다.
"""
import sys, math, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common.util import setup, save, alice_text, load_smollm

# ---------- 양자화 기본 부품
def q_sym(w, bits, dim=None, group=None):
    """대칭 absmax 양자화 → 역양자화. dim=None: 텐서 전체, dim=1: 행(출력 채널)별, group: 입력축 g개씩."""
    qmax = 2 ** (bits - 1) - 1
    if group:
        O, I = w.shape; wg = w.reshape(O, I // group, group)
        s = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / qmax
        return ((wg / s).round().clamp(-qmax - 1, qmax) * s).reshape(O, I)
    s = (w.abs().amax() if dim is None else w.abs().amax(dim, keepdim=True)).clamp(min=1e-8) / qmax
    return (w / s).round().clamp(-qmax - 1, qmax) * s


def gptq(w, X=None, bits=4, group=128, damp=0.01, H=None):
    """GPTQ(Frantar et al. 2022): 한 열씩 양자화하고, 그 오차를 아직 안 한 열에 Hessian으로 보정."""
    W = w.clone().double(); O, I = W.shape; qmax = 2 ** (bits - 1) - 1
    H = (2 * X.double().T @ X.double() / X.shape[0]) if H is None else H.clone()
    H += damp * H.diag().mean() * torch.eye(I, dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i in range(I):
        if i % group == 0:                                   # 그룹 시작: 지금(보정된) 값으로 scale
            s = W[:, i:i + group].abs().amax(1).clamp(min=1e-8) / qmax
        q = (W[:, i] / s).round().clamp(-qmax - 1, qmax) * s
        Q[:, i] = q
        err = (W[:, i] - q) / U[i, i]
        W[:, i + 1:] -= err.unsqueeze(1) * U[i, i + 1:].unsqueeze(0)   # OBS 보정
    return Q.float()


def awq(w, X, bits=4, group=128, grid=20):
    """AWQ(Lin et al. 2023): 활성값이 큰 입력 채널의 가중치를 s배 키워 양자화 → 다시 나눔."""
    xmean = X.abs().mean(0).clamp(min=1e-8); ref = X @ w.T
    best = (float("inf"), None, None)
    for a in torch.linspace(0, 1, grid + 1):
        s = xmean ** a; s = s / (s.max() * s.min()).sqrt()
        wq = q_sym(w * s, bits, group=group) / s
        e = ((X @ wq.T - ref) ** 2).mean().item()
        if e < best[0]:
            best = (e, wq, float(a))
    return best[1], best[2]


def q_act(x, bits=8, scale=None):
    qmax = 2 ** (bits - 1) - 1
    s = scale if scale is not None else x.abs().amax().clamp(min=1e-8) / qmax
    return (x / s).round().clamp(-qmax - 1, qmax) * s


def smooth_scales(X, w, alpha=0.5, xmax=None):
    """SmoothQuant: s_j = max|X_j|^α / max|W_j|^(1-α) — 난이도를 활성값→가중치로 옮김."""
    xmax = X.abs().amax(0) if xmax is None else xmax
    return (xmax.clamp(min=1e-5) ** alpha / w.abs().amax(0).clamp(min=1e-5) ** (1 - alpha))


def rel_err(y, ref):
    return (((y - ref) ** 2).sum() / (ref ** 2).sum()).item()


# ---------- 1) 합성 Linear: 아웃라이어 채널을 심는다
_B = torch.randn(128, 1024, generator=torch.Generator().manual_seed(9)) / 128 ** 0.5


def synth(n, I, outl, seed, scale=40.0):
    """실제 활성값처럼 '저차원 + 잡음' 구조(채널끼리 상관)를 갖게 만들고, 일부 채널을 크게."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 128, generator=g) @ _B + 0.3 * torch.randn(n, I, generator=g)
    X[:, outl] *= scale
    return X


def synthetic(R):
    I = O = 1024; G = 128
    g = torch.Generator().manual_seed(1)
    w = torch.randn(O, I, generator=g) * 0.02 * (1 + torch.rand(O, 1, generator=g) * 3)
    outl = torch.tensor([7, 100, 333, 512, 600, 801, 900, 1000])
    Xc, Xt = synth(4096, I, outl, 2), synth(2048, I, outl, 3)
    ref = Xt @ w.T
    print("\n[1] 합성 Linear 1024×1024, 아웃라이어 채널 8개(×40)")
    print(f"  활성값 채널별 max: 보통 채널 중앙값 {Xc.abs().amax(0).median():.1f}, 아웃라이어 {Xc[:, outl].abs().amax():.1f}")
    # 예측 vs 실측: 균일 양자화 오차 Δ²/12
    rows = []
    for name, kw in [("per-tensor", {}), ("per-channel", {"dim": 1}), ("group128", {"group": G})]:
        wq = q_sym(w, 4, **kw)
        if "group" in kw:
            s = w.reshape(O, I // G, G).abs().amax(-1) / 7; pred = (s ** 2 / 12).mean().item()
        elif "dim" in kw:
            s = w.abs().amax(1) / 7; pred = (s ** 2 / 12).mean().item()
        else:
            pred = ((w.abs().amax() / 7) ** 2 / 12).item()
        meas = ((wq - w) ** 2).mean().item()
        rows.append(dict(name=name, pred=pred, meas=meas))
        print(f"  W4 {name:11s} 가중치 MSE 예측 Δ²/12 = {pred:.3e}, 실측 {meas:.3e} (비 {meas/pred:.2f})")
    R["delta_sq"] = rows
    res = {}
    res["W4 per-tensor RTN"] = rel_err(Xt @ q_sym(w, 4).T, ref)
    res["W4 per-channel RTN"] = rel_err(Xt @ q_sym(w, 4, dim=1).T, ref)
    res["W4 g128 RTN"] = rel_err(Xt @ q_sym(w, 4, group=G).T, ref)
    res["W4 g128 GPTQ"] = rel_err(Xt @ gptq(w, Xc).T, ref)
    wa, alpha_awq = awq(w, Xc); res["W4 g128 AWQ"] = rel_err(Xt @ wa.T, ref)
    sc = Xc.abs().amax() / 127
    res["W8A8 naive"] = rel_err(q_act(Xt, scale=sc) @ q_sym(w, 8, dim=1).T, ref)
    s = smooth_scales(Xc, w)
    sc2 = (Xc / s).abs().amax() / 127
    res["W8A8 SmoothQuant"] = rel_err(q_act(Xt / s, scale=sc2) @ q_sym(w * s, 8, dim=1).T, ref)
    for k, v in res.items():
        print(f"  {k:20s} 출력 상대오차 {v:.2e}")
    print(f"  (AWQ가 고른 α = {alpha_awq:.2f})")
    R["synthetic"] = res; R["awq_alpha"] = alpha_awq
    R["act_absmax"] = Xc.abs().amax(0).tolist()
    # 실패 장면: 엉뚱한 calibration (아웃라이어 위치가 다른 데이터)
    wrong = torch.tensor([3, 50, 222, 444, 555, 777, 888, 999])
    Xw = synth(4096, I, wrong, 4)
    fail = {"W4 g128 RTN (calib 없음)": res["W4 g128 RTN"],
            "GPTQ, 맞는 calib": res["W4 g128 GPTQ"], "GPTQ, 틀린 calib": rel_err(Xt @ gptq(w, Xw).T, ref),
            "AWQ, 맞는 calib": res["W4 g128 AWQ"], "AWQ, 틀린 calib": rel_err(Xt @ awq(w, Xw)[0].T, ref),
            "SmoothQuant, 맞는 calib": res["W8A8 SmoothQuant"]}
    s_w = smooth_scales(Xw, w); scw = (Xw / s_w).abs().amax() / 127
    fail["SmoothQuant, 틀린 calib"] = rel_err(q_act(Xt / s_w, scale=scw) @ q_sym(w * s_w, 8, dim=1).T, ref)
    print("\n[실패 장면] calibration 데이터의 아웃라이어 위치가 실제와 다르면")
    for k, v in fail.items():
        print(f"  {k:24s} {v:.2e}")
    R["fail_calib"] = fail


# ---------- 2) SmolLM2-135M 실제 모델
def real(R):
    tok, model = load_smollm()
    text = alice_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    L = 256
    calib = ids[:32 * L].view(32, L)                          # 앞부분: calibration
    ev = ids[len(ids) // 2: len(ids) // 2 + 16 * L].view(16, L)   # 뒤쪽 절반: 평가
    g = torch.Generator().manual_seed(0)
    rand_calib = torch.randint(0, tok.vocab_size, (32, L), generator=g)
    lins = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "layers" in n}
    W0 = {n: m.weight.data.clone() for n, m in lins.items()}
    print(f"\n[2] SmolLM2-135M, 양자화 대상 Linear {len(lins)}개 (embedding/lm_head 제외), 평가 {ev.numel()} 토큰")

    @torch.no_grad()
    def ppl():
        nll = 0.0; n = 0
        for row in ev:
            out = model(row[None], labels=row[None]); nll += out.loss.item() * (L - 1); n += L - 1
        return math.exp(nll / n)

    @torch.no_grad()
    def collect(data):
        """Linear 입력의 통계만 모은다: H=2XᵀX/n (GPTQ), 채널별 max (SmoothQuant), 표본 512행 (AWQ 탐색)."""
        st = {n: dict(H=0, n=0, amax=0, sample=[]) for n in lins}; hs = []
        def hook(mod, inp, n):
            x = inp[0].reshape(-1, inp[0].shape[-1]).double(); d = st[n]
            d["H"] = d["H"] + 2 * x.T @ x; d["n"] += x.shape[0]
            d["amax"] = torch.maximum(torch.as_tensor(d["amax"], dtype=torch.float64), x.abs().amax(0))
            if len(d["sample"]) < 4: d["sample"].append(x[::64].float())
        for n, m in lins.items():
            hs.append(m.register_forward_pre_hook(lambda mod, inp, n=n: hook(mod, inp, n)))
        for row in data:
            model(row[None])
        for h in hs: h.remove()
        for d in st.values():
            d["H"] = d["H"] / d["n"]; d["amax"] = d["amax"].float(); d["sample"] = torch.cat(d["sample"])
        return st

    def restore():
        for n, m in lins.items(): m.weight.data.copy_(W0[n])

    out = {"FP32": ppl()}; print(f"  FP32            PPL {out['FP32']:.2f}")
    Xc = collect(calib)
    R["real_act_absmax"] = {n: Xc[n]["amax"].tolist() for n in
                            ["model.layers.3.self_attn.q_proj", "model.layers.3.mlp.down_proj"]}
    for name, fn in [("W4 per-channel RTN", lambda n, w: q_sym(w, 4, dim=1)),
                     ("W4 g64 RTN", lambda n, w: q_sym(w, 4, group=64)),
                     ("W4 g64 AWQ", lambda n, w: awq(w, Xc[n]["sample"], group=64, grid=10)[0]),
                     ("W4 g64 GPTQ", lambda n, w: gptq(w, group=64, H=Xc[n]["H"]))]:
        with torch.no_grad():
            for n, m in lins.items(): m.weight.data.copy_(fn(n, W0[n]))
        out[name] = ppl(); restore(); print(f"  {name:15s} PPL {out[name]:.2f}")
    # W8A8: 입력 활성값 per-tensor INT8 (정적 scale), 가중치 per-channel INT8
    for name, use_sq in [("W8A8 naive", False), ("W8A8 SmoothQuant", True)]:
        hs = []
        with torch.no_grad():
            for n, m in lins.items():
                s = smooth_scales(None, W0[n], xmax=Xc[n]["amax"]) if use_sq else torch.ones(W0[n].shape[1])
                m.weight.data.copy_(q_sym(W0[n] * s, 8, dim=1))
                sc = (Xc[n]["amax"] / s).amax() / 127
                hs.append(m.register_forward_pre_hook(lambda mod, inp, s=s, sc=sc: (q_act(inp[0] / s, scale=sc),)))
        out[name] = ppl(); restore()
        for h in hs: h.remove()
        print(f"  {name:15s} PPL {out[name]:.2f}")
    # 실패 장면: 랜덤 토큰으로 calibration
    Xr = collect(rand_calib)
    with torch.no_grad():
        for n, m in lins.items(): m.weight.data.copy_(gptq(W0[n], group=64, H=Xr[n]["H"]))
    out["W4 g64 GPTQ (랜덤 토큰 calib)"] = ppl(); restore()
    print(f"  GPTQ 랜덤 calib PPL {out['W4 g64 GPTQ (랜덤 토큰 calib)']:.2f}")
    hs = []
    with torch.no_grad():
        for n, m in lins.items():
            s = smooth_scales(None, W0[n], xmax=Xr[n]["amax"]); m.weight.data.copy_(q_sym(W0[n] * s, 8, dim=1))
            sc = (Xr[n]["amax"] / s).amax() / 127
            hs.append(m.register_forward_pre_hook(lambda mod, inp, s=s, sc=sc: (q_act(inp[0] / s, scale=sc),)))
    out["W8A8 SmoothQuant (랜덤 토큰 calib)"] = ppl(); restore()
    for h in hs: h.remove()
    print(f"  SmoothQuant 랜덤 calib PPL {out['W8A8 SmoothQuant (랜덤 토큰 calib)']:.2f}")
    R["real_ppl"] = out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--real", action="store_true"); a = ap.parse_args()
    R = {"env": setup(threads=1)}
    print("환경:", R["env"])
    print("[0] 실효 비트 = 4 + 16/g (fp16 scale만), zero-point까지 = 4 + 32/g")
    R["bits"] = {g: (4 + 16 / g, 4 + 32 / g) for g in (32, 64, 128, 256)}
    for g, (b1, b2) in R["bits"].items():
        print(f"  g={g:3d}: {b1:.3f} / {b2:.3f} bit")
    synthetic(R)
    if a.real:
        torch.set_num_threads(8); R["env_real_threads"] = 8
        real(R)
    save(Path(__file__).parent, "02" + ("_real" if a.real else ""), R)
