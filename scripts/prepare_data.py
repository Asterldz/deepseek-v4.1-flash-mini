"""Convert JSONL text/SFT or pretokenized records into mmap files, preserving boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer
from v41_mini.tokenization import engram_token_map, token_map_fingerprint
from v41_mini.integrity import DATA_FILES, sha256_file


def encoded_records(src, tokenizer, kind, eos, batch_size=256):
    """Yield records in order; batch ordinary pretraining text through Rayon."""
    pending = []

    def flush_pending():
        if not pending:
            return []
        encodings = tokenizer.encode_batch([item['text'] for item in pending], add_special_tokens=False)
        result = [(encoding.ids+[eos], encoding.ids+[eos]) for encoding in encodings]
        pending.clear()
        return result

    for line in src:
        item = json.loads(line)
        if kind == 'pretrain' and tokenizer is not None and 'input_ids' not in item:
            pending.append(item)
            if len(pending) >= batch_size:
                yield from flush_pending()
            continue
        yield from flush_pending()
        if 'input_ids' in item:
            ids = item['input_ids']
            if kind == 'sft' and 'labels' not in item:
                raise ValueError('Pretokenized SFT records require explicit labels with prompt masked -100')
            labels = item.get('labels', ids.copy())
        elif tokenizer is None:
            raise ValueError('Text records require --tokenizer')
        elif kind == 'pretrain':
            ids = tokenizer.encode(item['text'], add_special_tokens=False).ids+[eos]
            labels = ids.copy()
        else:
            prompt = tokenizer.encode('<user>'+item['prompt']+'<assistant>', add_special_tokens=False).ids
            answer = tokenizer.encode(item['response'], add_special_tokens=False).ids+[eos]
            ids, labels = prompt+answer, [-100]*len(prompt)+answer
        yield ids, labels
    yield from flush_pending()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, help='JSONL: text, or prompt/response, or input_ids/labels')
    p.add_argument('--output', required=True)
    p.add_argument('--tokenizer', help='tokenizers tokenizer.json; omitted only for input_ids records')
    p.add_argument('--kind', choices=['pretrain', 'sft'], required=True)
    p.add_argument('--vocab-size', type=int, default=32768)
    p.add_argument('--seq-len', type=int, default=2048)
    args = p.parse_args()
    if args.seq_len < 2 or args.vocab_size < 1:
        p.error('seq-len must be >=2 and vocab-size must be positive')
    tokenizer = Tokenizer.from_file(args.tokenizer) if args.tokenizer else None
    if tokenizer and tokenizer.get_vocab_size() != args.vocab_size:
        raise ValueError('Tokenizer vocab must exactly match model config')
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty; refusing to overwrite prepared data')
    out.mkdir(parents=True, exist_ok=True)
    offsets, skipped = [0], 0
    eos = tokenizer.token_to_id('<eos>') if tokenizer else None
    if tokenizer and eos is None:
        raise ValueError('Tokenizer must define <eos>')
    if tokenizer and args.kind == 'sft' and any(tokenizer.token_to_id(t) is None for t in ('<user>', '<assistant>')):
        raise ValueError('SFT tokenizer must define <user> and <assistant> boundaries')
    with (out/'tokens.bin').open('wb') as ids_file, (out/'labels.bin').open('wb') as labels_file, Path(args.input).open() as src:
        for ids, labels in encoded_records(src, tokenizer, args.kind, eos):
            if (not isinstance(ids, list) or not isinstance(labels, list)
                    or len(ids) != len(labels)
                    or any(type(t) is not int or not 0 <= t < args.vocab_size for t in ids)):
                raise ValueError('Invalid ids/labels')
            if any(type(t) is not int or (t != -100 and not 0 <= t < args.vocab_size) for t in labels):
                raise ValueError('Invalid label id')
            if args.kind == 'sft' and len(ids) > args.seq_len:
                skipped += 1
                continue
            # One-token overlap preserves next-token supervision across pretraining chunks.
            stride = args.seq_len-1
            if stride < 1:
                raise ValueError('seq-len must be >=2')
            starts = range(0, max(1, len(ids)-1), stride) if args.kind == 'pretrain' else [0]
            for start in starts:
                part, target = ids[start:start+args.seq_len], labels[start:start+args.seq_len]
                if len(part) < 2 or all(t == -100 for t in target[1:]):
                    skipped += 1
                    continue
                np.asarray(part, dtype='<u4').tofile(ids_file)
                np.asarray(target, dtype='<i4').tofile(labels_file)
                offsets.append(offsets[-1]+len(part))
    if len(offsets) == 1:
        raise ValueError('No usable records')
    np.save(out/'offsets.npy', np.asarray(offsets, dtype=np.int64))
    meta = dict(kind=args.kind, vocab_size=args.vocab_size, max_seq_len=args.seq_len,
                records=len(offsets)-1, stored_tokens=offsets[-1], skipped=skipped,
                source_sha256=sha256_file(args.input),
                files_sha256={name: sha256_file(out/name) for name in DATA_FILES},
                tokenizer_sha256=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest() if tokenizer else None,
                engram_token_map_sha256=token_map_fingerprint(engram_token_map(tokenizer)) if tokenizer else None)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
