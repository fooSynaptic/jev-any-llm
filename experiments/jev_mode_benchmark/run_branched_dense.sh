#!/usr/bin/env bash
# Dense HF: branched KV vs isolated wrap (Qwen3.5-4B / Qwen3.8-27B).
#
# Required env:
#   JEV_MODEL   — local checkpoint or hub id
#   JEV_DATA    — directory containing ag_news/test.csv
#   JEV_OUT     — output directory for JSON
#
# Optional:
#   JEV_BENCH_ROOT — this experiments/jev_mode_benchmark dir (default: script dir)
#   JEV_PER_CLASS, JEV_BATCH, JEV_LATENCY_CALLS, JEV_TAG
set -euo pipefail

MODEL="${JEV_MODEL:?set JEV_MODEL}"
DATA="${JEV_DATA:?set JEV_DATA}"
OUT="${JEV_OUT:?set JEV_OUT}"
ROOT="${JEV_BENCH_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
PER_CLASS="${JEV_PER_CLASS:-1000}"
BATCH="${JEV_BATCH:-16}"
LATENCY="${JEV_LATENCY_CALLS:-40}"
TAG="${JEV_TAG:-branched}"

mkdir -p "$OUT"
cd "$ROOT"

python3 benchmark_branched.py \
  --model "$MODEL" \
  --test-csv "$DATA/ag_news/test.csv" \
  --output "$OUT/${TAG}.json" \
  --per-class "$PER_CLASS" \
  --batch-size "$BATCH" \
  --latency-calls "$LATENCY" \
  "$@"

echo "wrote $OUT/${TAG}.json"
