#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/launch_4x4090.sh DATA_DIR TOKEN_MAP_JSON OUTPUT_DIR [extra train arguments]
if [ "$#" -lt 3 ]; then
  echo 'Usage: launch_4x4090.sh DATA_DIR TOKEN_MAP_JSON OUTPUT_DIR [extra train arguments]' >&2
  exit 2
fi
data_dir="$1"
token_map_file="$2"
output_dir="$3"
shift 3
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec torchrun --standalone --nnodes=1 --nproc-per-node=4 -m v41_mini.train \
  --config "$script_dir/../configs/v41_text_mini_500m.json" \
  --data "$data_dir" --token-map "$token_map_file" --output "$output_dir" \
  --device cuda --precision bf16 --zero1 --gradient-checkpointing \
  --seq-len 2048 --batch-size 1 --accumulation 8 \
  --optimizer hybrid --lr 0.0002 --engram-lr-scale 1 \
  --warmup-tokens 10000000 --decay-start-tokens 800000000 --total-tokens 1000000000 \
  --max-steps 500 --checkpoint-every 100 --keep-checkpoints 3 --verify-data --eval-every 100 "$@"
