"""Indexed token records; one record is an independent document or complete SFT example."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class TokenDataset(Dataset):
    def __init__(self, directory, seq_len, vocab_size):
        if seq_len < 2:
            raise ValueError('Training sequence length must be at least 2')
        self.directory = Path(directory)
        self.meta = json.loads((self.directory/'metadata.json').read_text())
        if self.meta['vocab_size'] != vocab_size:
            raise ValueError("Dataset and model vocab sizes do not match")
        if seq_len > self.meta['max_seq_len']:
            raise ValueError("Dataset records were prepared with a shorter sequence limit")
        self.ids = np.memmap(self.directory/'tokens.bin', dtype='<u4', mode='r')
        self.labels = np.memmap(self.directory/'labels.bin', dtype='<i4', mode='r')
        self.offsets = np.load(self.directory/'offsets.npy', mmap_mode='r')
        if (self.offsets.ndim != 1 or self.offsets.dtype.kind not in 'iu'
                or len(self.offsets) != self.meta['records']+1
                or len(self.ids) != len(self.labels)
                or len(self.ids) != self.meta['stored_tokens']
                or self.offsets[0] != 0 or self.offsets[-1] != len(self.ids)):
            raise ValueError('Prepared data files disagree with metadata/offsets')
        # Scan only the much smaller record index, never silently discard the suffix
        # of a pretraining document prepared for a longer context.
        for start in range(0, len(self.offsets)-1, 1_000_000):
            lengths = np.diff(self.offsets[start:min(start+1_000_001, len(self.offsets))])
            if (lengths < 2).any() or (lengths > seq_len).any():
                raise ValueError('Record length is invalid or exceeds training context; re-prepare data at the requested length')
        self.seq_len = seq_len

    def __len__(self):
        return len(self.offsets)-1

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        start, end = map(int, self.offsets[index:index+2])
        ids = torch.from_numpy(np.array(self.ids[start:end], dtype=np.int64))
        labels = torch.from_numpy(np.array(self.labels[start:end], dtype=np.int64))
        return dict(input_ids=ids, labels=labels)


def collate_records(records):
    length = max(len(r['input_ids']) for r in records)
    ids = torch.zeros(len(records), length, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    segments = torch.full_like(ids, -1)
    for i, r in enumerate(records):
        n = len(r['input_ids'])
        ids[i, :n], labels[i, :n], segments[i, :n] = r['input_ids'], r['labels'], 0
    return dict(input_ids=ids, labels=labels, segment_ids=segments)


class SyntheticDataset(Dataset):
    """Deterministic arithmetic token pattern for pipeline smoke tests, not language data."""
    def __init__(self, seq_len, vocab_size, size=256):
        self.seq_len, self.vocab_size, self.size = seq_len, vocab_size, size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ids = (torch.arange(self.seq_len)+index) % (self.vocab_size-4)+4
        return dict(input_ids=ids.long(), labels=ids.long().clone())
