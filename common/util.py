"""공용 도우미: 경로, 스레드 고정, 환경 기록, 결과 JSON 저장, 공개 텍스트 받기."""
import json, os, platform, sys, urllib.request
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def setup(threads=1, seed=0):
    torch.set_num_threads(threads); torch.manual_seed(seed)
    return env()


def env():
    return dict(python=platform.python_version(), torch=torch.__version__,
                threads=torch.get_num_threads(), cpu=platform.processor() or platform.machine())


def save(chapter_dir, name, obj):
    out = Path(chapter_dir) / "results"; out.mkdir(exist_ok=True)
    p = out / f"{name}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    print(f"[저장] {p.relative_to(ROOT)}")


def alice_text():
    """Project Gutenberg #11 (퍼블릭 도메인) 『이상한 나라의 앨리스』 본문. 한 번만 받아서 캐시."""
    DATA.mkdir(exist_ok=True); p = DATA / "alice.txt"
    if not p.exists():
        urllib.request.urlretrieve("https://www.gutenberg.org/cache/epub/11/pg11.txt", p)
    t = p.read_text(encoding="utf-8")
    s = t.find("CHAPTER I."); s = t.find("CHAPTER I.", s + 10)   # 목차 다음 본문 시작
    e = t.find("*** END OF THE PROJECT GUTENBERG")
    return t[s:e]


def load_smollm(dtype=torch.float32, attn=None):
    """--real 예제용 SmolLM2-135M (약 270MB, Apache-2.0)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "HuggingFaceTB/SmolLM2-135M"
    tok = AutoTokenizer.from_pretrained(name)
    kw = {"attn_implementation": attn} if attn else {}
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, **kw).eval()
    return tok, model
