import json
import subprocess
import sys
import numpy as np
import pytest
from v41_mini.data import TokenDataset


def prepare(tmp_path, item, kind='pretrain'):
    source = tmp_path/'source.jsonl'
    source.write_text(json.dumps(item)+'\n')
    return subprocess.run([sys.executable, 'scripts/prepare_data.py', '--input', str(source),
                           '--output', str(tmp_path/'data'), '--kind', kind,
                           '--vocab-size', '128', '--seq-len', '8'],
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('item', [dict(input_ids=[4, 5.5, 6]),
                                 dict(input_ids=[4, True, 6]),
                                 dict(input_ids=[4, 5, 6], labels=[4, 5.5, 6])])
def test_pretokenized_noninteger_ids_are_not_silently_cast(tmp_path, item):
    result = prepare(tmp_path, item)
    assert result.returncode != 0
    assert 'Invalid' in result.stderr


def test_pretraining_context_change_requires_repacking(tmp_path):
    result = prepare(tmp_path, dict(input_ids=list(range(12))))
    assert result.returncode == 0, result.stderr
    data = TokenDataset(tmp_path/'data', 8, 128)
    # Overlap retains every adjacent pair across chunk boundaries exactly once.
    pairs = [pair for record in data for pair in zip(record['input_ids'].tolist(), record['input_ids'][1:].tolist())]
    assert pairs == list(zip(range(11), range(1, 12)))
    with pytest.raises(ValueError, match='re-prepare'):
        TokenDataset(tmp_path/'data', 4, 128)


def test_corrupt_record_offsets_fail_before_training(tmp_path):
    result = prepare(tmp_path, dict(input_ids=list(range(12))))
    assert result.returncode == 0, result.stderr
    np.save(tmp_path/'data/offsets.npy', np.array([0, 5, 200]))
    with pytest.raises(ValueError, match='metadata/offsets'):
        TokenDataset(tmp_path/'data', 8, 128)
