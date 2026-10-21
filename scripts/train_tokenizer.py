import argparse
import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from v41_mini.tokenization import engram_token_map


def main():
    p = argparse.ArgumentParser(description='Train lossless byte-level BPE and a separate Engram normalization map')
    p.add_argument('--input', nargs='+', required=True, help='UTF-8 plain text files, not JSONL')
    p.add_argument('--output', required=True)
    p.add_argument('--vocab-size', type=int, default=32768)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    special = ['<pad>', '<bos>', '<eos>', '<draft>', '<user>', '<assistant>', '<think>', '</think>']
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, special_tokens=special,
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train(args.input, trainer)
    if tok.get_vocab_size() != args.vocab_size:
        raise ValueError('Corpus too small for requested vocab; add data or explicitly change the model vocab')
    tok.save(str(out/'tokenizer.json'))
    mapping = engram_token_map(tok)
    (out/'engram_token_map.json').write_text(json.dumps(mapping)+'\n')
    print(f'Saved tokenizer and Engram map; vocab={tok.get_vocab_size()} compressed_ids={len(set(mapping))}')


if __name__ == '__main__':
    main()
