import json
from pathlib import Path
import subprocess
import sys
import torch
import pytest


def run_training(path, steps, resume=None, config='tiny'):
    cmd = [sys.executable, '-m', 'v41_mini.train', '--config', config, '--synthetic',
           '--output', str(path), '--seq-len', '12', '--max-steps', str(steps), '--threads', '2',
           '--gradient-checkpointing', '--accumulation', '2']
    if resume:
        cmd += ['--resume', str(resume)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout+'\n'+result.stderr
    return torch.load(path/'last.pt', weights_only=False)


@pytest.mark.parametrize('config', ['tiny', 'tiny-24'])
def test_training_resume_matches_uninterrupted_run(tmp_path, config):
    expected = run_training(tmp_path/'full', 4, config=config)
    run_training(tmp_path/'split', 2, config=config)
    actual = run_training(tmp_path/'split', 4, tmp_path/'split/last.pt', config=config)
    assert expected['progress'] == actual['progress']
    for key, value in expected['model'].items():
        torch.testing.assert_close(value, actual['model'][key], rtol=0, atol=0, msg=key)
    for key, values in expected['optimizer']['state'].items():
        for field, value in values.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, actual['optimizer']['state'][key][field], rtol=0, atol=0)


def test_prepare_sft_preserves_mask_and_rejects_long_answers(tmp_path):
    source = tmp_path/'source.jsonl'
    source.write_text('\n'.join(json.dumps(x) for x in [
        dict(input_ids=[1,2,3,4,5], labels=[-100,-100,-100,4,5]),
        dict(input_ids=list(range(20)), labels=list(range(20)))])+'\n')
    result = subprocess.run([sys.executable, 'scripts/prepare_data.py', '--input', str(source),
                             '--output', str(tmp_path/'data'), '--kind', 'sft', '--vocab-size', '128',
                             '--seq-len', '8'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    from v41_mini.data import TokenDataset, collate_records
    ds = TokenDataset(tmp_path/'data', 8, 128)
    assert len(ds) == 1
    batch = collate_records([ds[0]])
    assert batch['labels'].tolist() == [[-100,-100,-100,4,5]]
    assert ds.meta['skipped'] == 1
