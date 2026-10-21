#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/launch_sft_4gpu.sh BASE_PT SFT_DATA_DIR TOKEN_MAP_JSON OUTPUT_DIR [extra arguments]
if [ "$#" -lt 4 ]; then
  echo 'Usage: launch_sft_4gpu.sh BASE_PT SFT_DATA_DIR TOKEN_MAP_JSON OUTPUT_DIR [extra arguments]' >&2
  exit 2
fi
base_checkpoint="$1"
data_dir="$2"
token_map_file="$3"
output_dir="$4"
shift 4
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage_args=(--config "$script_dir/../configs/v41_text_mini_500m.json")
resume_requested=0
for arg in "$@"; do
  case "$arg" in
    --resume|--resume=*) resume_requested=1; break ;;
  esac
done
if [ "$resume_requested" -eq 0 ]; then
  stage_args+=(--init-from "$base_checkpoint")
fi
exec torchrun --standalone --nnodes=1 --nproc-per-node=4 -m v41_mini.train \
  "${stage_args[@]}" \
  --data "$data_dir" --token-map "$token_map_file" --output "$output_dir" \
  --device cuda --precision bf16 --zero1 --gradient-checkpointing \
  --seq-len 2048 --batch-size 1 --accumulation 4 \
  --optimizer hybrid --lr 0.00001 \
  --warmup-tokens 1000000 --decay-start-tokens 50000000 --total-tokens 100000000 \
  --max-steps 500 --checkpoint-every 100 --keep-checkpoints 3 --verify-data "$@"
