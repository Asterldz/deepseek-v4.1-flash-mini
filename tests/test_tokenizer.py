import json
import subprocess
import sys
from dataclasses import replace
from tokenizers import Tokenizer
from v41_mini import ModelConfig
from v41_mini.tokenization import token_map_fingerprint


def test_tokenizer_roundtrip_and_separate_engram_map(tmp_path):
    corpus = tmp_path/'corpus.txt'
    corpus.write_text(('中文数学：若 X + 3 = 7，则 X = 4。\nPython: x = 1.25; SQL SELECT A FROM t;\n'
                       'LaTeX: \\frac{a}{b}  α + β ≤ 10\n')*64)
    out = tmp_path/'tokenizer'
    result = subprocess.run([sys.executable, 'scripts/train_tokenizer.py', '--input', str(corpus),
                             '--output', str(out), '--vocab-size', '280'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    tok = Tokenizer.from_file(str(out/'tokenizer.json'))
    text = '若 X² − x = 0，\n  则 x ∈ {0, 1}；答案：\\frac{1}{2}。🙂'
    assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text
    mapping = json.loads((out/'engram_token_map.json').read_text())
    assert len(mapping) == 280 and all(i >= 0 for i in mapping)
    assert tok.token_to_id('<draft>') == 3
    source = tmp_path/'source.jsonl'
    source.write_text(json.dumps({'text': text})+'\n')
    data = tmp_path/'data'
    result = subprocess.run([sys.executable, 'scripts/prepare_data.py', '--input', str(source),
                             '--output', str(data), '--kind', 'pretrain', '--seq-len', '32',
                             '--vocab-size', '280', '--tokenizer', str(out/'tokenizer.json')],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((data/'metadata.json').read_text())['engram_token_map_sha256'] == token_map_fingerprint(mapping)

    config = tmp_path/'tiny.json'
    config.write_text(json.dumps(replace(ModelConfig.tiny(), vocab_size=280).to_dict()))
    command = [sys.executable, '-m', 'v41_mini.train', '--config', str(config), '--data', str(data),
               '--seq-len', '32', '--max-steps', '1', '--threads', '1']
    wrong_map = tmp_path/'wrong_map.json'
    wrong_map.write_text(json.dumps([(mapping[0]+1) % 280]+mapping[1:]))
    rejected = subprocess.run(command+['--token-map', str(wrong_map), '--output', str(tmp_path/'rejected')],
                              capture_output=True, text=True, timeout=30)
    assert rejected.returncode != 0 and 'does not match the tokenizer' in rejected.stderr
    accepted = subprocess.run(command+['--token-map', str(out/'engram_token_map.json'),
                                        '--output', str(tmp_path/'accepted')],
                              capture_output=True, text=True, timeout=60)
    assert accepted.returncode == 0, accepted.stdout+'\n'+accepted.stderr
    assert (tmp_path/'accepted/last.pt').is_file()
