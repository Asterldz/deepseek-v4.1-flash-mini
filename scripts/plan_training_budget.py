"""Calculate a provisional token budget from real aggregate training step measurements."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def plan(rows, hours=20, reserve_hours=4, warmup_steps=20):
    rows = [row for row in rows if row.get('step', 0) > warmup_steps and 'aggregate_tokens_per_second' in row]
    if len(rows) < 20:
        raise ValueError('Need at least 20 measured updates after warmup; CPU/tiny results cannot predict server performance')
    speeds = np.array([row['aggregate_tokens_per_second'] for row in rows], dtype=float)
    if not np.isfinite(speeds).all() or (speeds <= 0).any():
        raise ValueError('Invalid throughput measurements')
    if not 0 < reserve_hours < hours:
        raise ValueError('Require 0 < reserve-hours < hours')
    speed = float(np.percentile(speeds, 25))
    # Reserve time for validation, save, setup and fluctuations; still not a guarantee.
    tokens = math.floor(speed*(hours-reserve_hours)*3600/1_000_000)*1_000_000
    if tokens < 1_000_000:
        raise ValueError('Measured budget is below 1M tokens')
    return dict(status='provisional_requires_same_model_hardware_context_and_data_lengths',
                total_gpu_wall_hours=hours, reserved_hours=reserve_hours,
                measured_updates=len(rows), aggregate_tokens_per_second_p25=speed,
                total_tokens=tokens, warmup_tokens=tokens//100, decay_start_tokens=tokens*8//10,
                note='Freeze this schedule BEFORE the one real-data run; do not modify it during --resume. Synthetic throughput can overestimate real-data throughput.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metrics', required=True)
    p.add_argument('--hours', type=float, default=20)
    p.add_argument('--reserve-hours', type=float, default=4)
    p.add_argument('--warmup-steps', type=int, default=20)
    args = p.parse_args()
    rows = [json.loads(line) for line in Path(args.metrics).read_text().splitlines() if line.strip()]
    print(json.dumps(plan(rows, args.hours, args.reserve_hours, args.warmup_steps), indent=2))


if __name__ == '__main__':
    main()
