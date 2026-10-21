import argparse
import hashlib
from pathlib import Path
import torch
from tokenizers import Tokenizer
from .config import ModelConfig
from .model import TextMini


def main():
    p = argparse.ArgumentParser(description='Text generation with full or approximate CED prefill')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--prefill', choices=['full', 'bounded_replay'], default='full')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
    p.add_argument('--chat', action='store_true')
    args = p.parse_args()
    if args.max_new_tokens < 0:
        p.error('--max-new-tokens must be nonnegative')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ModelConfig(**ckpt['config'])
    model = TextMini(config)
    model.load_state_dict(ckpt['model'])
    model.to(device=args.device, dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float32).eval()
    tok = Tokenizer.from_file(args.tokenizer)
    if tok.get_vocab_size() != config.vocab_size:
        raise ValueError('Tokenizer/model vocabulary mismatch')
    expected_hash = (ckpt.get('data_identity', {}).get('metadata') or {}).get('tokenizer_sha256')
    if expected_hash and hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest() != expected_hash:
        raise ValueError('Tokenizer content differs from the tokenizer used for training')
    # Training checkpoints also contain optimizer states, unnecessary during generation.
    del ckpt
    text = '<user>'+args.prompt+'<assistant>' if args.chat else args.prompt
    ids = torch.tensor([tok.encode(text, add_special_tokens=False).ids], device=args.device)
    if ids.shape[1] == 0:
        raise ValueError('Prompt must encode to at least one token')
    if ids.shape[1]+args.max_new_tokens > config.max_seq_len:
        raise ValueError('Prompt plus requested output exceeds trained context')
    result = model.generate(ids, args.max_new_tokens, args.temperature, tok.token_to_id('<eos>'), args.prefill)
    print(tok.decode(result[0, ids.shape[1]:].tolist(), skip_special_tokens=False))


if __name__ == '__main__':
    main()
