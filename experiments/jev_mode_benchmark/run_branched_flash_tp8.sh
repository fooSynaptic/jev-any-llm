#!/usr/bin/env bash
# DeepSeek-V4.1-Flash TP8: branched KV vs isolated wrap.
#
# Required env:
#   JEV_FLASH_CKPT      — TP8 shard dir (model{0..7}-mp8.safetensors)
#   JEV_FLASH_HF        — upstream HF tree (inference/ + encoding/)
#   JEV_DATA            — directory containing ag_news/test.csv
#   JEV_OUT             — output directory for JSON
#
# Optional:
#   JEV_BENCH_ROOT, JEV_FLASH_CONFIG, JEV_NPROC, JEV_MASTER_PORT,
#   JEV_PER_CLASS, JEV_FWD_BATCH, JEV_LATENCY_CALLS, JEV_BRANCHED_CAP, JEV_TAG
set -euo pipefail

CKPT="${JEV_FLASH_CKPT:?set JEV_FLASH_CKPT}"
HF="${JEV_FLASH_HF:?set JEV_FLASH_HF}"
DATA="${JEV_DATA:?set JEV_DATA}"
OUT="${JEV_OUT:?set JEV_OUT}"
ROOT="${JEV_BENCH_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
CONFIG="${JEV_FLASH_CONFIG:-$HF/inference/config.json}"
NPROC="${JEV_NPROC:-8}"
PORT="${JEV_MASTER_PORT:-29581}"
PER_CLASS="${JEV_PER_CLASS:-500}"
FWD_BATCH="${JEV_FWD_BATCH:-4}"
LATENCY="${JEV_LATENCY_CALLS:-24}"
CAP="${JEV_BRANCHED_CAP:-200}"
TAG="${JEV_TAG:-branched_flash_v41}"

if [[ -n "${JEV_FLASH_PYTHON:-}" ]]; then
  export PATH="$(dirname "$JEV_FLASH_PYTHON"):$PATH"
fi

mkdir -p "$OUT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$HF/inference"

torchrun --nproc-per-node "$NPROC" --master_port "$PORT" \
  "$ROOT/benchmark_branched_flash_tp8.py" \
  --ckpt-path "$CKPT" \
  --config "$CONFIG" \
  --tokenizer-path "$CKPT" \
  --inference-dir "$HF/inference" \
  --encoding-dir "$HF/encoding" \
  --test-csv "$DATA/ag_news/test.csv" \
  --output "$OUT/${TAG}.json" \
  --per-class "$PER_CLASS" \
  --fwd-batch "$FWD_BATCH" \
  --latency-calls "$LATENCY" \
  --branched-quality-cap "$CAP" \
  "$@"

echo "wrote $OUT/${TAG}.json"
