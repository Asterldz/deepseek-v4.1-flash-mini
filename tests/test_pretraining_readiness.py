import json
from pathlib import Path
import subprocess
import sys
import random
import numpy as np
import pytest
import torch
from scripts.audit_corpus import audit, quality_flags
from scripts.plan_training_budget import plan
from v41_mini.data import TokenDataset
from v41_mini.integrity import DATA_FILES, sha256_file, verify_dataset


def make_data(path):
    path.mkdir()
    np.array([4, 5, 6, 7, 8, 9], dtype='<u4').tofile(path/'tokens.bin')
    np.array([4, 5, 6, 7, 8, 9], dtype='<i4').tofile(path/'labels.bin')
    np.save(path/'offsets.npy', np.array([0, 3, 6], dtype=np.int64))
    meta = dict(kind='pretrain', vocab_size=128, max_seq_len=8, records=2, stored_tokens=6,
                files_sha256={name: sha256_file(path/name) for name in DATA_FILES})
    (path/'metadata.json').write_text(json.dumps(meta))
    return meta


def test_integrity_detects_same_size_corruption(tmp_path):
    make_data(tmp_path/'data')
    before = verify_dataset(TokenDataset(tmp_path/'data', 8, 128))
    assert before['supervised_targets'] == 4
    np.array([4, 5, 6, 7, 8, 10], dtype='<u4').tofile(tmp_path/'data/tokens.bin')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_dataset(TokenDataset(tmp_path/'data', 8, 128))


def test_integrity_checks_labels_even_with_valid_hashes(tmp_path):
    meta = make_data(tmp_path/'data')
    np.array([4, 5, 6, 7, 8, 10], dtype='<i4').tofile(tmp_path/'data/labels.bin')
    meta['files_sha256']['labels.bin'] = sha256_file(tmp_path/'data/labels.bin')
    (tmp_path/'data/metadata.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='unshifted'):
        verify_dataset(TokenDataset(tmp_path/'data', 8, 128))


def record(text, group):
    return dict(text=text, source='fixture', source_id=group, group_id=group, revision='abc', license='test')


def test_audit_detects_duplicate_and_family_leakage(tmp_path):
    train, heldout = tmp_path/'train.jsonl', tmp_path/'heldout.jsonl'
    heldout.write_text(json.dumps(record('A mathematical problem.', 'original-task'))+'\n')
    train.write_text('\n'.join(json.dumps(r) for r in [
        record('A  mathematical\nproblem.', 'copied-task'),
        record('同一道题的中文翻译。', 'original-task'),
        record('Another document.', 'independent'),
        record('Another document.', 'duplicate'),
    ])+'\n')
    report = audit([train], [heldout], tmp_path/'audit')
    assert report['blockers']['heldout_exact_overlap'] == 1
    assert report['blockers']['heldout_group_overlap'] == 1
    assert report['blockers']['normalized_duplicate'] == 1
    assert report['ready_for_training'] is False


def test_audit_does_not_claim_semantic_quality(tmp_path):
    train, heldout = tmp_path/'train.jsonl', tmp_path/'heldout.jsonl'
    train.write_text(json.dumps(record('Training text.', 'train'))+'\n')
    heldout.write_text(json.dumps(record('Validation text.', 'val'))+'\n')
    report = audit([train], [heldout], tmp_path/'audit')
    assert not report['blockers']
    assert report['status'] == 'mechanical_checks_passed_review_required'
    assert report['ready_for_training'] is False
    assert quality_flags('a\x00b')


def test_budget_uses_aggregate_speed_and_reserves_time():
    rows = [dict(step=i, aggregate_tokens_per_second=2000) for i in range(1, 61)]
    result = plan(rows)
    assert result['total_tokens'] == 115_000_000
    assert result['warmup_tokens'] < result['decay_start_tokens'] < result['total_tokens']
    with pytest.raises(ValueError, match='20 measured'):
        plan(rows[:25])


def test_checkpoints_keep_recent_and_latest_can_resume(tmp_path):
    out = tmp_path/'train'
    command = [sys.executable, '-m', 'v41_mini.train', '--config', 'tiny-24', '--synthetic',
               '--device', 'cpu', '--output', str(out), '--seq-len', '8', '--threads', '2',
               '--checkpoint-every', '1', '--keep-checkpoints', '2']
    result = subprocess.run(command+['--max-steps', '4'], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.glob('step-*.pt')) == ['step-0000003.pt', 'step-0000004.pt']
    latest = torch.load(out/'latest.pt', weights_only=False)
    assert latest['progress']['step'] == 4
    result = subprocess.run(command+['--max-steps', '5', '--resume', str(out/'latest.pt')],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert sorted(p.name for p in out.glob('step-*.pt')) == ['step-0000004.pt', 'step-0000005.pt']
    assert torch.load(out/'latest.pt', weights_only=False)['progress']['step'] == 5
    metrics = [json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert len(metrics[-1]['router_entropy_per_layer']) == 24


def test_wall_budget_survives_resume_without_more_updates(tmp_path):
    out = tmp_path/'train'
    command = [sys.executable, '-m', 'v41_mini.train', '--config', 'tiny-24', '--synthetic',
               '--device', 'cpu', '--output', str(out), '--seq-len', '8', '--threads', '2']
    result = subprocess.run(command+['--max-steps', '1'], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    before = torch.load(out/'last.pt', weights_only=False)
    assert before['wall_seconds'] > 0
    result = subprocess.run(command+['--max-steps', '2', '--resume', str(out/'last.pt'),
                                    '--max-wall-hours', str(before['wall_seconds']/3600)],
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    after = torch.load(out/'last.pt', weights_only=False)
    assert after['progress']['step'] == 1
    assert after['wall_seconds'] > before['wall_seconds']
    assert 'wall_budget_stop' in result.stdout


def test_preview_samples_cannot_pass_as_training_data(tmp_path):
    source = tmp_path/'preview.jsonl'
    source.write_text(json.dumps(dict(record('Example text.', 'a'), preview_only=True,
                                     license=['cc-by-sa-3.0', 'gfdl']))+'\n')
    result = audit([source], [], tmp_path/'audit')
    assert result['blockers']['preview_only_not_training_data'] == 1
    assert 'missing_license' not in result['blockers']


def test_exact_math_generator_checks_inverse_relations_and_rejects_wrong_answers():
    from scripts.make_verified_math import make_example, verify_example
    rng = random.Random(18)
    tasks = set()
    for _ in range(100):
        item = make_example(rng)
        tasks.add(item['task'])
        verify_example(item)
        with pytest.raises(ValueError, match='verification'):
            verify_example(dict(item, answer='999999999'))
    assert tasks == {'add', 'subtract', 'multiply', 'divide', 'linear'}
