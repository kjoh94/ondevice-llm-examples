# 온디바이스 LLM 시리즈 예제 코드

블로그 [온디바이스 LLM 시리즈 — 목차](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-series/)의 장별 예제입니다.
모든 `example.py`는 **CPU만으로** 돌아가고, 기본 실행은 다운로드가 없습니다. `--real` 옵션을 붙인 장만 공개 모델
[SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)(약 270MB, Apache-2.0)과 퍼블릭 도메인 텍스트
(Project Gutenberg #11)를 받습니다.

```bash
pip install -r requirements.txt            # 기본
pip install -r requirements-real.txt       # --real 옵션용
cd 01-roofline && python example.py
./run_all.sh                               # 전 장 실행 (REAL=1 이면 --real까지)
```

| 장 | 폴더 | 블로그 |
|---|---|---|
| 1 | [`01-roofline`](01-roofline/) | [80 TOPS인데 왜 30 tok/s일까: Roofline과 메모리 대역폭](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-01-roofline-bandwidth/) |
| 2 | [`02-quantization`](02-quantization/) | [4비트로 줄이고도 답을 지키는 법: LLM 양자화 실전 (W4A16·W8A8·GPTQ·AWQ·SmoothQuant)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-02-llm-quantization/) |
| 3 | [`03-kv-cache`](03-kv-cache/) | [메모장이 가중치보다 커질 때: KV 캐시 설계 (GQA·정적 캐시·KV 양자화·sink)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-03-kv-cache-design/) |
| 4 | [`04-op-split`](04-op-split/) | [어떤 연산을 어디서 돌릴까: NPU·CPU·DSP 연산 분할 (Amdahl·경계 비용·비선형 근사)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-04-npu-cpu-dsp-split/) |
| 5 | [`05-graph-export`](05-graph-export/) | [PyTorch 모델을 NPU용 그래프로: prefill/decode 2그래프 export (torch.export·ONNX·ORT)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-05-llm-graph-export/) |
| 6 | [`06-gguf`](06-gguf/) | [llama.cpp와 GGUF 해부: 파일 한 개에 모델이 들어가는 방식 (Q8_0·Q4_0·Q4_K·llama-bench)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-06-llamacpp-gguf/) |
| 7 | [`07-bench`](07-bench/) | [TTFT·TPS를 제대로 재는 법: 같은 모델이 20 tok/s도 277 tok/s도 되는 이유](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-07-ttft-tps-benchmark/) |
| 8 | [`08-speculative`](08-speculative/) | [추측해서 한 번에 검증하기: Speculative decoding (수락률·γ·KV 되감기·prompt lookup)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-08-speculative-decoding/) |
| 9 | [`09-moe-pim`](09-moe-pim/) | [큰 모델을 작은 대역폭으로: MoE와 메모리 안에서 계산하기(PIM)](https://kjoh94.github.io/blog/2026/11/16/ondevice-llm-09-moe-pim/) |

공용 코드
- [`common/tiny_llama.py`](common/tiny_llama.py): RMSNorm·RoPE·GQA·SwiGLU·정적 KV 캐시를 갖춘 미니 Llama (3·4·5·7장).
- [`common/roofline.py`](common/roofline.py): `roofline_calc()` — 본문의 모든 상한 숫자는 이 함수에서 나옵니다.
- [`common/util.py`](common/util.py): 스레드·시드 고정, 결과 JSON 저장(`NN-*/results/*.json`), 텍스트·모델 로드.

6장 실제 GGUF와 `llama-bench`는 [`tools/make_gguf_and_bench.sh`](tools/make_gguf_and_bench.sh)로 만듭니다
(llama.cpp 공식 릴리스 바이너리 + `convert_hf_to_gguf.py` 필요, 예제 실행 필수 조건은 아님).

## 검증 환경

블로그 숫자는 모두 이 저장소 코드를 아래 환경에서 실제로 돌려 얻은 값입니다(`*/results/*.json`).
Intel Xeon(Sapphire Rapids급) 가상머신 8 vCPU, L3 320 MiB, Python 3.13.5, PyTorch 2.15.0+cpu, onnxruntime 1.32.0,
transformers 5.23.0, gguf 0.19.0, llama.cpp b12428. 시간 측정은 별도 표기가 없으면 **CPU 1스레드**입니다.

## 공개 정보 원칙

이 저장소와 블로그의 "엣지 NPU"는 **공개된 일반 스펙 수준의 숫자로 만든 가상의 장치**입니다
(80 TOPS, LPDDR5X 128bit 153.6 GB/s, 약 5 W). 특정 회사 제품의 내부 정보나 실측치는 쓰지 않았고,
정밀도처럼 확인할 수 없는 값은 코드와 본문에 `가정`으로 표시했습니다.
