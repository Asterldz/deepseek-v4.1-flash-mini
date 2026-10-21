"""Streaming checks for immutable prepared training artifacts."""
import hashlib
from pathlib import Path
import numpy as np


DATA_FILES = ('tokens.bin', 'labels.bin', 'offsets.npy')


def sha256_file(path):
    with Path(path).open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def verify_dataset(dataset):
    expected = dataset.meta.get('files_sha256')
    if not isinstance(expected, dict) or set(expected) != set(DATA_FILES):
        raise ValueError('Missing prepared-file checksums; re-prepare data before verified training')
    actual = {name: sha256_file(dataset.directory/name) for name in DATA_FILES}
    if actual != expected:
        raise ValueError('Prepared data checksum mismatch; do not resume with changed files')
    supervised = 0
    for start in range(0, len(dataset.ids), 1_000_000):
        ids, labels = dataset.ids[start:start+1_000_000], dataset.labels[start:start+1_000_000]
        if (ids >= dataset.meta['vocab_size']).any():
            raise ValueError('Out-of-vocabulary token in prepared data')
        if ((labels != -100) & ((labels < 0) | (labels >= dataset.meta['vocab_size']))).any():
            raise ValueError('Invalid prepared label')
        if dataset.meta['kind'] == 'pretrain' and not np.array_equal(ids, labels):
            raise ValueError('Pretraining labels must equal unshifted input IDs')
        supervised += int((labels != -100).sum())
    # Every independent record's first label is outside the next-token objective.
    supervised -= int((dataset.labels[dataset.offsets[:-1]] != -100).sum())
    if supervised <= 0:
        raise ValueError('Dataset contains no supervised next-token targets')
    lengths = np.diff(dataset.offsets)
    return dict(files_sha256=actual, records=len(dataset), stored_tokens=len(dataset.ids),
                supervised_targets=supervised, mean_record_tokens=float(lengths.mean()),
                record_length_percentiles=dict(zip(('p10', 'p50', 'p90', 'p99'),
                                                  map(float, np.percentile(lengths, [10, 50, 90, 99])))))
