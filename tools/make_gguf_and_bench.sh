#!/usr/bin/env bash
# SmolLM2-135M → GGUF(F16/Q8_0/Q4_0/Q4_K_M) 변환 후 llama-bench pp512/tg128 (1·8스레드, 3회)
set -e; cd "$(dirname "$0")"  # llama-b12428/ (릴리스 바이너리)와 llama.cpp-b12428/ (소스)를 이 폴더에 풀어 두세요
B=llama-b12428; SRC=llama.cpp-b12428; PY=${PY:-python}
SNAP=$(ls -d ~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/*)
[ -f smollm2-135m-f16.gguf ] || $PY $SRC/convert_hf_to_gguf.py "$SNAP" --outtype f16 --outfile smollm2-135m-f16.gguf
for q in Q8_0 Q4_0 Q4_K_M; do [ -f smollm2-135m-$q.gguf ] || $B/llama-quantize smollm2-135m-f16.gguf smollm2-135m-$q.gguf $q; done
$B/llama-bench -m smollm2-135m-f16.gguf,smollm2-135m-Q8_0.gguf,smollm2-135m-Q4_0.gguf,smollm2-135m-Q4_K_M.gguf \
  -t 1,8 -p 512 -n 128 -r 3 -o json > bench.json 2> bench.err
