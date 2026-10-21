"""Shared Engram normalization and a canonical fingerprint for tokenizer binding."""
import hashlib
import json
import unicodedata


def engram_token_map(tokenizer):
    normalized, mapping = {}, []
    special_ids = {i for i, token in tokenizer.get_added_tokens_decoder().items() if token.special}
    for i in range(tokenizer.get_vocab_size()):
        raw = tokenizer.id_to_token(i)
        text = tokenizer.decode([i], skip_special_tokens=False)
        if '\ufffd' in text or i in special_ids:
            key = ('raw', raw)
        else:
            text = unicodedata.normalize('NFD', unicodedata.normalize('NFKC', text)).lower()
            text = ''.join(ch for ch in text if not unicodedata.combining(ch))
            key = ('normalized', ' '.join(text.split()))
        if key not in normalized:
            normalized[key] = len(normalized)
        mapping.append(normalized[key])
    return mapping


def token_map_fingerprint(mapping):
    return hashlib.sha256(json.dumps(mapping, separators=(',', ':')).encode('utf-8')).hexdigest()
