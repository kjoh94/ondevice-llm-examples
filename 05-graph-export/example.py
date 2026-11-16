"""5장 예제 — 모양이 매번 바뀌는 그래프를 NPU에: prefill/decode 2그래프 export → ONNX → ORT
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-05-llm-graph-export/

python example.py   (torch, onnx, onnxscript, onnxruntime / 다운로드 없음)
"""
import sys, time, random, warnings, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn as nn
from common.util import setup, save
from common.tiny_llama import Config, TinyLlama, apply_rope, greedy

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent; OUT = HERE / "onnx"; OUT.mkdir(exist_ok=True)


class StepGraph(nn.Module):
    """KV 캐시를 그래프 입력·출력으로 드러낸 한 스텝. shape: ids (1,T), pos (1,), K/V (L,1,Hkv,S,hd)."""
    def __init__(self, m: TinyLlama, use_pos=True):
        super().__init__(); self.m = m; self.use_pos = use_pos

    def forward(self, ids, pos, k_cache, v_cache):
        m, c = self.m, self.m.cfg; T = ids.shape[1]; hd = c.head_dim
        positions = (pos + torch.arange(T)) if self.use_pos else torch.arange(T)   # 실패 장면: use_pos=False
        cos, sin = m.cos[positions], m.sin[positions]
        kpos = torch.arange(c.max_seq).unsqueeze(0)
        mask = torch.where(kpos <= positions.unsqueeze(1), 0.0, float("-inf"))
        x = m.emb(ids); ks, vs = [], []
        for l, blk in enumerate(m.layers):
            a = blk.attn; h = blk.n1(x)
            q = a.q(h).view(1, T, c.n_heads, hd).transpose(1, 2)
            k = a.k(h).view(1, T, c.n_kv_heads, hd).transpose(1, 2)
            v = a.v(h).view(1, T, c.n_kv_heads, hd).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            kc = k_cache[l].index_copy(2, positions, k)          # 새 텐서 반환(out-of-place)
            vc = v_cache[l].index_copy(2, positions, v)
            ks.append(kc); vs.append(vc)
            rep = c.n_heads // c.n_kv_heads
            kk, vv = kc.repeat_interleave(rep, 1), vc.repeat_interleave(rep, 1)
            att = ((q @ kk.transpose(-1, -2)) / hd ** 0.5 + mask).softmax(-1) @ vv
            x = x + a.o(att.transpose(1, 2).reshape(1, T, -1))
            h = blk.n2(x)
            x = x + blk.down(torch.nn.functional.silu(blk.gate(h)) * blk.up(h))
        return m.head(m.norm(x)), torch.stack(ks), torch.stack(vs)


def empty_cache(c):
    return torch.zeros(c.n_layers, 1, c.n_kv_heads, c.max_seq, c.head_dim)


def run_generate(step_prefill, step_decode, prompt, n_new, bucket, c, to_np=False):
    """prompt를 bucket 길이로 오른쪽 패딩 → prefill 1번 → decode n_new-1번."""
    P = prompt.shape[1]
    ids = torch.zeros(1, bucket, dtype=torch.long); ids[:, :P] = prompt
    K, V = empty_cache(c), empty_cache(c)
    logits, K, V = step_prefill(ids, torch.tensor([0]), K, V)
    out = [int(np.argmax(logits[0, P - 1]))]; pos = P; ts = []
    for _ in range(n_new - 1):
        t0 = time.perf_counter()
        logits, K, V = step_decode(torch.tensor([[out[-1]]]), torch.tensor([pos]), K, V)
        ts.append(time.perf_counter() - t0)
        out.append(int(np.argmax(logits[0, -1]))); pos += 1
    return out, ts


def ort_fn(path):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    def f(ids, pos, K, V):
        feeds = dict(zip(names, [t.numpy() if isinstance(t, torch.Tensor) else t for t in (ids, pos, K, V)]))
        lo, k2, v2 = sess.run(None, feeds)
        return lo, k2, v2
    return f, sess


if __name__ == "__main__":
    import onnx
    R = {"env": setup(threads=1)}; print("환경:", R["env"])
    cfg = Config(vocab=2048, dim=256, n_layers=4, n_heads=8, n_kv_heads=2, ffn=704, max_seq=128)
    model = TinyLlama(cfg).eval(); g = StepGraph(model).eval()
    BUCKET = 32
    prompt = torch.randint(0, cfg.vocab, (1, 20), generator=torch.Generator().manual_seed(3))
    ref = greedy(model, prompt, 20, "static")
    # ---------- 1) PyTorch에서 2그래프 방식이 원래 모델과 같은가
    with torch.no_grad():
        eager, ts_eager = run_generate(g, g, prompt, 20, BUCKET, cfg)
    print(f"[1] 패딩 버킷 {BUCKET} + 정적 캐시 2그래프(eager) == 원래 greedy: {eager == ref}")
    # ---------- 2) torch.export → ONNX
    K0, V0 = empty_cache(cfg), empty_cache(cfg)
    ex_args = {"prefill": (torch.zeros(1, BUCKET, dtype=torch.long), torch.tensor([0]), K0, V0),
               "decode": (torch.zeros(1, 1, dtype=torch.long), torch.tensor([5]), K0, V0)}
    sizes = {}
    for name, args in ex_args.items():
        ep = torch.export.export(g, args)
        n_nodes = len(ep.graph.nodes)
        path = OUT / f"{name}.onnx"
        onnx_prog = torch.onnx.export(g, args, dynamo=True, input_names=["input_ids", "pos", "k_cache", "v_cache"],
                                      output_names=["logits", "k_out", "v_out"], optimize=True, verbose=False)
        onnx_prog.save(str(path))
        mp = onnx.load(str(path)); ops = {}
        for n in mp.graph.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
        sizes[name] = dict(fx_nodes=n_nodes, onnx_nodes=len(mp.graph.node), mb=os.path.getsize(path) / 1e6,
                           top_ops=sorted(ops.items(), key=lambda x: -x[1])[:10])
        print(f"[2] {name}: ExportedProgram 노드 {n_nodes}, ONNX 노드 {len(mp.graph.node)}, 파일 {sizes[name]['mb']:.2f} MB")
        print("    ONNX op 상위:", sizes[name]["top_ops"])
    R["export"] = sizes
    # ---------- 3) ORT로 20토큰 생성 → 토큰 완전 일치 + 수치 차이
    fp, sp = ort_fn(OUT / "prefill.onnx"); fd, sd = ort_fn(OUT / "decode.onnx")
    ort_out, ts_ort = run_generate(fp, fd, prompt, 20, BUCKET, cfg)
    with torch.no_grad():
        lt, _, _ = g(*ex_args["prefill"][:1], torch.tensor([0]), K0, V0)
    lo, _, _ = fp(ex_args["prefill"][0], torch.tensor([0]), K0, V0)
    diff = float(np.abs(lo - lt.numpy()).max())
    t_e, t_o = np.median(ts_eager) * 1e3, np.median(ts_ort) * 1e3
    print(f"[3] ORT 20토큰 == PyTorch: {ort_out == ref}, prefill logits 최대 차이 {diff:.2e}")
    print(f"    decode 1토큰 중앙값: PyTorch eager {t_e:.2f} ms, ORT {t_o:.2f} ms")
    R["ort"] = dict(same=ort_out == ref, maxdiff=diff, eager_ms=t_e, ort_ms=t_o, tokens=ort_out[:10])
    # ---------- 4) 동적 shape: torch.cat 캐시 버전을 export하면?
    class CatAttn(nn.Module):
        def forward(self, q, k_new, v_new, k_past, v_past):
            k = torch.cat([k_past, k_new], 2); v = torch.cat([v_past, v_new], 2)
            return (q @ k.transpose(-1, -2)).softmax(-1) @ v, k, v
    ca = CatAttn(); H, hd = 2, 32
    mk = lambda T, S: (torch.randn(1, H, T, hd), torch.randn(1, H, T, hd), torch.randn(1, H, T, hd),
                       torch.randn(1, H, S, hd), torch.randn(1, H, S, hd))
    logs = {}
    ep_static = torch.export.export(ca, mk(1, 10))
    try:
        ep_static.module()(*mk(1, 11)); logs["no_dim"] = "성공(?)"
    except Exception as e:
        logs["no_dim"] = type(e).__name__ + ": " + str(e).splitlines()[0][:160]
    print("[4] Dim 선언 없이 export(past=10) 후 past=11로 호출 →", logs["no_dim"])
    past = torch.export.Dim("past", min=1, max=4096)
    ds = ({}, {}, {}, {2: past}, {2: past})
    ep_dyn = torch.export.export(ca, mk(1, 10), dynamic_shapes=ds)
    y = ep_dyn.module()(*mk(1, 777))
    logs["with_dim"] = f"성공: past=777 → k_out {tuple(y[1].shape)}; range_constraints {list(ep_dyn.range_constraints.values())[:1]}"
    print("    Dim('past', 1..4096) 선언 →", logs["with_dim"])
    try:
        ep_dyn.module()(*mk(1, 5000)); logs["out_of_range"] = "성공(?)"
    except Exception as e:
        logs["out_of_range"] = type(e).__name__ + ": " + str(e).splitlines()[0][:160]
    print("    past=5000(max 초과) →", logs["out_of_range"])
    R["dynamic"] = logs
    # ---------- 5) 버킷 패딩 낭비: 식 vs 시뮬레이션
    random.seed(0)
    lens = [random.randint(1, 1024) for _ in range(100000)]
    sim = {}
    for name, buckets in {"{1024}": [1024], "{128,512}": [128, 512, 1024], "{128,256,512,1024}": [128, 256, 512, 1024],
                          "64 간격": list(range(64, 1025, 64))}.items():
        pad = tot = 0
        for L in lens:
            b = next(x for x in buckets if x >= L); pad += b - L; tot += b
        # 식: 길이가 (a,b] 균등이면 평균 패딩 ≈ (b-a-1)/2, 확률 ∝ (b-a)
        edges = [0] + buckets; pred_pad = pred_tot = 0
        for a, b in zip(edges[:-1], edges[1:]):
            w = (b - a) / 1024; pred_pad += w * (b - a - 1) / 2; pred_tot += w * b
        sim[name] = dict(meas=pad / tot, pred=pred_pad / pred_tot)
        print(f"[5] 버킷 {name:20s} 낭비율 식 {pred_pad/pred_tot:.1%} / 시뮬레이션 {pad/tot:.1%}")
    R["bucket"] = sim
    # ---------- 실패 장면: position_ids(=pos)를 빼먹은 decode 그래프
    bad = StepGraph(model, use_pos=False).eval()
    with torch.no_grad():
        wrong, _ = run_generate(g, bad, prompt, 20, BUCKET, cfg)
    first = next((i for i, (a, b) in enumerate(zip(wrong, ref)) if a != b), None)
    print(f"[실패 장면] decode에서 위치를 0부터 다시 세면: 일치 {sum(a == b for a, b in zip(wrong, ref))}/20, 처음 틀린 위치 {first}")
    print("   정답 :", ref[:12]); print("   버그 :", wrong[:12])
    R["fail_pos"] = dict(ref=ref, wrong=wrong, first=first)
    save(HERE, "05", R)
