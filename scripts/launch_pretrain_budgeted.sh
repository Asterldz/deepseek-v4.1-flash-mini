#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 5 ]; then
  echo 'Usage: launch_pretrain_budgeted.sh DATA_DIR VALIDATION_DIR TOKEN_MAP_JSON OUTPUT_DIR BUDGET_JSON [extra train arguments]' >&2
  exit 2
fi
data_dir="$1"
validation_dir="$2"
token_map_file="$3"
output_dir="$4"
budget_file="$5"
shift 5
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
budget_values="$("$python_bin" - "$budget_file" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
keys = ('total_tokens', 'warmup_tokens', 'decay_start_tokens')
if any(type(b.get(k)) is not int for k in keys):
    raise ValueError('Budget must contain integer token schedule values')
if not 0 <= b['warmup_tokens'] <= b['decay_start_tokens'] < b['total_tokens']:
    raise ValueError('Invalid token schedule')
if b.get('total_gpu_wall_hours') != 20 or b.get('reserved_hours', 0) < 4:
    raise ValueError('This launcher requires the selected 20-hour budget with at least 4 hours reserved')
print(*(b[k] for k in keys))
PY
)"
read -r total_tokens warmup_tokens decay_start_tokens <<< "$budget_values"
exec bash "$script_dir/launch_4x4090.sh" "$data_dir" "$token_map_file" "$output_dir" \
  --validation-data "$validation_dir" --total-tokens "$total_tokens" \
  --warmup-tokens "$warmup_tokens" --decay-start-tokens "$decay_start_tokens" \
  --max-steps 2147483647 --max-wall-hours 18 "$@"
