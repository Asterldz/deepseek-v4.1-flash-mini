import json
import os
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.mark.parametrize('resume', [[], ['--resume', 'sft/last.pt'], ['--resume=sft/last.pt']])
def test_sft_launcher_resumes_without_reinitializing(tmp_path, resume):
    # Capture the command without starting distributed workers or allocating a model.
    fake = tmp_path/'torchrun'
    fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    script = Path('scripts/launch_sft_4gpu.sh').resolve()
    result = subprocess.run(['bash', str(script), 'base.pt', 'data', 'map.json', 'out', *resume],
                            cwd=tmp_path, env={**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert ('--init-from' in args) == (not resume)
    assert Path(args[args.index('--config')+1]).is_file()
    assert Path(args[args.index('--config')+1]).name == 'v41_text_mini_500m.json'
    assert '--nproc-per-node=4' in args
    if resume:
        assert args[-len(resume):] == resume


def test_pretraining_launcher_uses_compact_four_card_budget(tmp_path):
    fake = tmp_path/'torchrun'
    fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    script = Path('scripts/launch_4x4090.sh').resolve()
    result = subprocess.run(['bash', str(script), 'data', 'map.json', 'out'], cwd=tmp_path,
                            env={**os.environ, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    assert '--nproc-per-node=4' in args
    config = json.loads(Path(args[args.index('--config')+1]).read_text())
    assert (config['layers'], config['dim'], config['expert_dim']) == (24, 640, 288)
    assert (config['routed_experts'], config['topk_experts']) == (32, 4)
    assert args[args.index('--precision')+1] == 'bf16'
    assert 4*int(args[args.index('--batch-size')+1])*int(args[args.index('--accumulation')+1]) == 32
    assert [int(args[args.index(flag)+1]) for flag in ('--warmup-tokens', '--decay-start-tokens', '--total-tokens')] == [10_000_000, 800_000_000, 1_000_000_000]
    assert args[args.index('--max-steps')+1] == '500'


def test_budgeted_launcher_applies_measured_schedule_and_wall_cap(tmp_path):
    fake = tmp_path/'torchrun'
    fake.write_text(f'#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n')
    fake.chmod(0o755)
    budget = tmp_path/'budget with spaces.json'
    budget.write_text(json.dumps(dict(total_gpu_wall_hours=20, reserved_hours=4,
                                     total_tokens=115000000, warmup_tokens=1150000, decay_start_tokens=92000000)))
    script = Path('scripts/launch_pretrain_budgeted.sh').resolve()
    env = {**os.environ, 'PYTHON_BIN': sys.executable, 'PATH': str(tmp_path)+os.pathsep+os.environ['PATH']}
    command = ['bash', str(script), 'data', 'val', 'map.json', 'out', str(budget)]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    args = json.loads(result.stdout)
    def last_value(flag):
        return args[max(i for i, v in enumerate(args) if v == flag)+1]
    assert last_value('--total-tokens') == '115000000'
    assert last_value('--max-wall-hours') == '18'
    assert last_value('--validation-data') == 'val'
    assert '--verify-data' in args and '--nproc-per-node=4' in args
    budget.write_text('{}')
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and not result.stdout
