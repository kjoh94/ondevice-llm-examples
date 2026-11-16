"""6장 예제 — 온디바이스 표준 런타임 뜯어보기: GGUF 직접 파싱, Q8_0/Q4_0/Q4_K 역양자화
블로그: https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-06-llamacpp-gguf/

python example.py                         # 작은 GGUF를 만들고 직접 파싱 (다운로드 없음)
python example.py --gguf model-Q4_K_M.gguf # 실제 GGUF(예: SmolLM2-135M 변환본)의 텐서 형식·Q4_K 검증
"""
import sys, struct, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import gguf
from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType as QT, quants
from common.util import save

HERE = Path(__file__).parent
BLOCK = {"Q8_0": (32, 2 + 32), "Q4_0": (32, 2 + 16), "Q4_K": (256, 2 + 2 + 12 + 128),
         "Q6_K": (256, 2 + 16 + 192), "Q5_0": (32, 2 + 4 + 16), "F16": (1, 2), "F32": (1, 4)}


def bpw(t):
    n, b = BLOCK[t]; return 8 * b / n


# ---------- 1) 작은 GGUF 만들기
def write_demo(path):
    rng = np.random.default_rng(0)
    w = GGUFWriter(str(path), arch="demo")
    w.add_name("ondevice-llm-series demo"); w.add_uint32("demo.block_count", 2)
    w.add_float32("demo.rope.freq_base", 10000.0); w.add_array("demo.tags", ["gguf", "q8_0", "q4_0"])
    t = {"tok_embd.weight": rng.standard_normal((16, 64)).astype(np.float32),
         "blk.0.ffn_up.weight": rng.standard_normal((64, 256)).astype(np.float32),
         "blk.0.ffn_down.weight": rng.standard_normal((256, 64)).astype(np.float32)}
    w.add_tensor("tok_embd.weight", t["tok_embd.weight"])
    w.add_tensor("blk.0.ffn_up.weight", quants.quantize(t["blk.0.ffn_up.weight"], QT.Q8_0), raw_dtype=QT.Q8_0)
    w.add_tensor("blk.0.ffn_down.weight", quants.quantize(t["blk.0.ffn_down.weight"], QT.Q4_0), raw_dtype=QT.Q4_0)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return t


# ---------- 2) 순수 파이썬 파서 (명세: ggml/docs/gguf.md)
VT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def parse(buf, respect_alignment=True):
    off = 0
    def rd(fmt):
        nonlocal off; v = struct.unpack_from(fmt, buf, off); off += struct.calcsize(fmt); return v[0]
    def rstr():
        nonlocal off; n = rd("<Q"); s = buf[off:off + n].decode(); off += n; return s
    def rval(t):
        if t == 8: return rstr()
        if t == 9:
            et = rd("<I"); n = rd("<Q"); return [rval(et) for _ in range(n)]
        return rd(VT[t])
    magic = buf[:4]; off = 4
    ver = rd("<I"); n_t = rd("<Q"); n_kv = rd("<Q")
    kv = {}
    for _ in range(n_kv):
        k = rstr(); t = rd("<I"); kv[k] = rval(t)
    tinfo = []
    for _ in range(n_t):
        name = rstr(); nd = rd("<I"); dims = [rd("<Q") for _ in range(nd)]; typ = rd("<I"); o = rd("<Q")
        tinfo.append(dict(name=name, dims=dims, type=QT(typ).name, offset=o))
    align = kv.get("general.alignment", 32)
    data_start = (off + align - 1) // align * align if respect_alignment else off
    return dict(magic=magic, version=ver, n_tensors=n_t, kv=kv, tensors=tinfo,
                header_end=off, align=align, data_start=data_start)


# ---------- 3) 역양자화 직접 구현 (ggml-common.h 블록 구조)
def deq_q8_0(raw):
    b = raw.reshape(-1, 34)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    q = b[:, 2:].copy().view(np.int8).astype(np.float32)
    return (d * q).reshape(-1)


def deq_q4_0(raw):
    b = raw.reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    qs = b[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8; hi = (qs >> 4).astype(np.int8) - 8   # 앞 16개=하위 니블, 뒤 16개=상위
    return (d * np.concatenate([lo, hi], 1).astype(np.float32)).reshape(-1)


def deq_q4_k(raw):
    b = raw.reshape(-1, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc_raw = b[:, 4:16].astype(np.int32); qs = b[:, 16:]
    sc = np.zeros((len(b), 8), np.int32); mn = np.zeros((len(b), 8), np.int32)
    for j in range(8):                                    # 6비트 scale/min 8쌍을 12바이트에 압축
        if j < 4:
            sc[:, j] = sc_raw[:, j] & 63; mn[:, j] = sc_raw[:, j + 4] & 63
        else:
            sc[:, j] = (sc_raw[:, j + 4] & 0xF) | ((sc_raw[:, j - 4] >> 6) << 4)
            mn[:, j] = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
    out = np.zeros((len(b), 256), np.float32)
    for c in range(4):                                    # 64개 단위: 32바이트의 하위/상위 니블
        q = qs[:, 32 * c:32 * c + 32]
        for h, nib in enumerate(((q & 0xF), (q >> 4))):
            j = 2 * c + h
            out[:, 64 * c + 32 * h:64 * c + 32 * h + 32] = d * sc[:, j:j + 1] * nib - dmin * mn[:, j:j + 1]
    return out.reshape(-1)


DEQ = {"Q8_0": deq_q8_0, "Q4_0": deq_q4_0, "Q4_K": deq_q4_k}


def tensor_bytes(info):
    n = int(np.prod(info["dims"])); k, b = BLOCK[info["type"]]
    return n // k * b


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--gguf", nargs="*", default=[]); a = ap.parse_args()
    R = {"gguf_py": getattr(gguf, "__version__", "0.19.0")}
    print("[0] 블록 형식별 실효 비트(bpw):", {k: round(bpw(k), 4) for k in BLOCK})
    R["bpw"] = {k: bpw(k) for k in BLOCK}
    path = HERE / "demo.gguf"; orig = write_demo(path)
    buf = path.read_bytes()
    print(f"\n[1] demo.gguf {len(buf)} 바이트, 앞 48바이트:")
    for i in range(0, 48, 16):
        chunk = buf[i:i + 16]
        print(f"  {i:08x}  {' '.join(f'{x:02x}' for x in chunk):47s}  {''.join(chr(x) if 32 <= x < 127 else '.' for x in chunk)}")
    R["hexdump"] = buf[:48].hex()
    p = parse(buf)
    print(f"[2] magic {p['magic']}, version {p['version']}, 텐서 {p['n_tensors']}개, KV {len(p['kv'])}개, "
          f"헤더 끝 {p['header_end']} → 정렬({p['align']}) 후 데이터 시작 {p['data_start']}")
    for k, v in p["kv"].items(): print(f"  KV {k} = {v}")
    rd = GGUFReader(str(path)); ok_off = True
    for ti, rt in zip(p["tensors"], rd.tensors):
        mine = p["data_start"] + ti["offset"]
        ok_off &= mine == rt.data_offset
        print(f"  텐서 {ti['name']:22s} {ti['type']:5s} dims {ti['dims']}  내 오프셋 {mine}  GGUFReader {rt.data_offset}")
    R["parse"] = dict(header_end=p["header_end"], data_start=p["data_start"], align=p["align"], offsets_match=ok_off,
                      kv={k: str(v) for k, v in p["kv"].items()}, tensors=p["tensors"])
    print(f"  오프셋 전부 일치: {ok_off}")
    # 역양자화: 내 구현 vs gguf 패키지 (비트 단위)
    res = {}
    for ti in p["tensors"]:
        if ti["type"] in DEQ:
            s = p["data_start"] + ti["offset"]; raw = np.frombuffer(buf, np.uint8, tensor_bytes(ti), s)
            mine = DEQ[ti["type"]](raw)
            ref = quants.dequantize(raw.reshape(1, -1), QT[ti["type"]]).reshape(-1)
            o = orig[ti["name"]].reshape(-1)
            rms = float(np.sqrt(((mine - o) ** 2).mean())); amax = float(np.abs(o).max())
            res[ti["name"]] = dict(type=ti["type"], bit_exact=bool(np.array_equal(mine, ref)), rmse=rms)
            print(f"[3] {ti['type']} 직접 역양자화 == gguf.quants: {np.array_equal(mine, ref)}, 원본 대비 RMSE {rms:.4f}")
    R["dequant"] = res
    # 실패 장면: 정렬 무시
    pb = parse(buf, respect_alignment=False)
    ti = pb["tensors"][1]; s = pb["data_start"] + ti["offset"]
    bad = deq_q8_0(np.frombuffer(buf, np.uint8, tensor_bytes(ti), s))
    good_s = p["data_start"] + ti["offset"]
    good = deq_q8_0(np.frombuffer(buf, np.uint8, tensor_bytes(ti), good_s))
    print(f"\n[실패 장면] 정렬 무시: 데이터 시작 {pb['data_start']} (정답 {p['data_start']}, {p['data_start']-pb['data_start']}바이트 밀림)")
    print(f"  ffn_up 앞 4개 정상 {[round(float(x), 3) for x in good[:4]]} / 밀림 {[round(float(x), 1) for x in bad[:4]]}, 최대값 {np.nanmax(np.abs(bad)):.3g}")
    R["fail_align"] = dict(shift=p["data_start"] - pb["data_start"], good=good[:4].tolist(), bad=[float(x) for x in bad[:4]],
                           bad_max=float(np.nanmax(np.abs(bad))))
    # ---------- 4) 실제 GGUF 파일
    real = {}
    for f in a.gguf:
        rd = GGUFReader(f); types = {}; tot_bytes = 0; tot_n = 0
        for t in rd.tensors:
            n = int(np.prod(t.shape)); name = t.tensor_type.name
            types.setdefault(name, [0, 0]); types[name][0] += 1; types[name][1] += n
            tot_bytes += int(t.n_bytes); tot_n += n
        fsize = Path(f).stat().st_size
        q4k = next((t for t in rd.tensors if t.tensor_type == QT.Q4_K), None)
        exact = None
        if q4k is not None:
            raw = np.asarray(q4k.data).reshape(-1).view(np.uint8)
            exact = bool(np.array_equal(deq_q4_k(raw), quants.dequantize(raw.reshape(1, -1), QT.Q4_K).reshape(-1)))
        real[Path(f).name] = dict(file_MB=fsize / 1e6, tensor_MB=tot_bytes / 1e6, n_params=tot_n,
                                  avg_bpw=8 * tot_bytes / tot_n, types=types, q4k_exact=exact)
        print(f"\n[4] {Path(f).name}: 파일 {fsize/1e6:.1f} MB, 파라미터 {tot_n/1e6:.1f}M, 평균 {8*tot_bytes/tot_n:.2f} bpw"
              + (f", 내 Q4_K 역양자화 == gguf: {exact}" if exact is not None else ""))
        for k, (c, n) in sorted(types.items(), key=lambda x: -x[1][1]):
            print(f"    {k:5s} 텐서 {c:3d}개, 원소 {n/1e6:6.2f}M")
    R["real"] = real
    save(HERE, "06", R)
