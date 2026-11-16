#!/usr/bin/env bash
# 모든 장 예제를 차례로 실행하고 로그를 logs/에 남긴다. (다른 무거운 작업이 없는 상태에서 실행할 것)
set -e
cd "$(dirname "$0")"; mkdir -p logs
PY=${PY:-python}
run() { echo "== $*"; $PY -u "$@" 2>&1 | grep -v "Loading weights" | tee "logs/$(dirname "$1")$( [ "$2" ] && echo _real ).log"; }
run 01-roofline/example.py
run 02-quantization/example.py
run 03-kv-cache/example.py
run 04-op-split/example.py
run 05-graph-export/example.py
run 06-gguf/example.py
run 07-bench/example.py
run 08-speculative/example.py
run 09-moe-pim/example.py
if [ "$REAL" = 1 ]; then
  run 02-quantization/example.py --real
  run 03-kv-cache/example.py --real
  run 08-speculative/example.py --real
  # 7장: llama-bench 결과(tools/bench.json)가 있으면 같이 정리
  if [ -f "${BENCH_JSON:-../tools/bench.json}" ]; then
    $PY -u 07-bench/example.py --real --llama-bench "${BENCH_JSON:-../tools/bench.json}" 2>&1 | grep -v "Loading weights" | tee logs/07-bench_real.log
  else
    run 07-bench/example.py --real
  fi
  if [ -n "$GGUF_DIR" ]; then
    $PY -u 06-gguf/example.py --gguf "$GGUF_DIR"/*.gguf 2>&1 | tee logs/06-gguf_real.log
  fi
fi
