import copy
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.make_zh_code_tools import PROGRAMS, build, episode, execute, sft_records, verify
from scripts.curate_code_tool_candidates import convert, normalize_hermes


@pytest.mark.parametrize('task', ['sum', 'filter', 'group', 'sort', 'dedup', 'mean', 'sql', 'repair', 'csv'])
def test_generated_tool_trajectory_replays_and_detects_tampered_output(task):
    rows = [dict(订单号='A', 类别='食品', 金额=10), dict(订单号='B', 类别='食品', 金额=-3),
            dict(订单号='A', 类别='办公', 金额=10)]
    item = episode(task, rows, 9, 'fixture')
    verify(item)
    records = list(sft_records(item))
    assert records
    # The first training call must not see a future tool result.
    if task != 'repair':
        assert '"role":"tool"' not in records[0]['prompt']
    else:
        assert all(PROGRAMS['bad_column'] not in json.dumps(json.loads(r['response'])) for r in records)
        assert 'KeyError' in records[0]['prompt']
    corrupted = copy.deepcopy(item)
    next(m for m in corrupted['messages'] if m['role'] == 'tool')['content'] = {'ok': True, 'result': 999}
    with pytest.raises(ValueError, match='does not match execution'):
        verify(corrupted)


def test_fixture_executor_rejects_arbitrary_code_and_queries():
    with pytest.raises(ValueError, match='authored Python'):
        execute('python_data', {'code': '__import__("os").system("echo bad")', 'data': {}})
    with pytest.raises(ValueError, match='authored read-only'):
        execute('sqlite_select', {'query': 'DROP TABLE sales', 'rows': []})
    item = episode('mean', [dict(订单号='A', 类别='办公', 金额=None)], 0, 'nulls')
    assert item['expected'] is None
    verify(item)


def test_generated_splits_keep_all_table_variants_together(tmp_path):
    root = tmp_path/'corpus'
    result = build(root, 50, seed=8)
    groups = {}
    for split in ('train', 'validation', 'test'):
        records = [json.loads(line) for line in (root/f'{split}.episodes.jsonl').read_text().splitlines()]
        groups[split] = {r['group_id'] for r in records}
        assert len(records) == len(groups[split])*9
    assert not groups['train'] & groups['validation']
    assert not groups['train'] & groups['test']
    assert not groups['validation'] & groups['test']
    assert sum(v['episodes'] for v in result['splits'].values()) == 450


def test_public_python_filter_checks_syntax_without_execution():
    base = dict(lang='python', problem='Sort a list of numbers', solution='```python\ndef f(xs):\n return sorted(xs)\n```')
    assert convert(base, 'magicoder')['verification'].endswith('NOT_execution_verified')
    with pytest.raises(ValueError, match='parseable'):
        convert(dict(base, solution='```python\ndef broken(:\n```'), 'magicoder')


def test_public_tool_filter_rejects_missing_argument_and_orphan_response():
    tool = dict(type='function', function=dict(name='sum_values', parameters=dict(type='object', properties=dict(values=dict(type='array')), required=['values'])))
    row = dict(tools=[tool], conversations=[
        dict(from_='human', value='Sum these values'),
        dict(from_='gpt', value='<tool_call>{"name":"sum_values","arguments":{"values":[1,2]}}</tool_call>'),
        dict(from_='tool', value='<tool_response>{"name":"sum_values","content":3}</tool_response>'),
        dict(from_='gpt', value='The result is 3.')])
    for turn in row['conversations']:
        turn['from'] = turn.pop('from_')
    assert len(normalize_hermes(row)[1]) == 4
    assert normalize_hermes(dict(row, tools=json.dumps([tool]))) == normalize_hermes(row)
    bad = copy.deepcopy(row)
    bad['conversations'][1]['value'] = '<tool_call>{"name":"sum_values","arguments":{}}</tool_call>'
    with pytest.raises(ValueError, match='missing_required'):
        normalize_hermes(bad)
    bad = copy.deepcopy(row)
    bad['conversations'][2]['value'] = '<tool_response>{"name":"other","content":3}</tool_response>'
    with pytest.raises(ValueError, match='mismatched'):
        normalize_hermes(bad)


def test_public_downloader_rejects_truncated_payload(tmp_path, monkeypatch):
    from scripts.fetch_code_tool_data import fetch
    source = dict(name='example', repo='test/data', revision='a'*40, file='data.jsonl',
                  expected_bytes=100, expected_sha256='0'*64)
    monkeypatch.setattr('urllib.request.urlopen', lambda *args, **kwargs: io.BytesIO(b'{"a":1}\n'))
    with pytest.raises(ValueError, match='possibly truncated'):
        fetch(source, tmp_path)
    assert not (tmp_path/'example.jsonl').exists()
    assert not list(tmp_path.glob('*.partial'))


def test_public_curation_deduplicates_shared_conversation_prefixes(tmp_path):
    import hashlib
    from scripts.curate_code_tool_candidates import curate
    raw = tmp_path/'raw'
    raw.mkdir()
    prefix = [dict(role='user', content='Sort these Python values'),
              dict(role='assistant', content='```python\nresult = sorted([2, 1])\n```')]
    rows = [dict(id=i, messages=prefix+[dict(role='user', content=f'Explain step {i}'),
                                      dict(role='assistant', content=f'Explanation {i}')]) for i in (1, 2)]
    source = raw/'code_feedback.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (raw/'manifest.json').write_text(json.dumps(dict(sources=[dict(name='code_feedback', repo='fixture',
        revision='a'*40, license_label='test', files=[dict(path=source.name, sha256=hashlib.sha256(source.read_bytes()).hexdigest())])])))
    report = curate(raw, tmp_path/'out')
    assert sum(s['episodes'] for s in report['splits'].values()) == 2
    assert sum(s['sft_records'] for s in report['splits'].values()) == 3
    assert sum(s['duplicate_sft_records_removed'] for s in report['splits'].values()) == 1


def test_chinese_sft_export_prepares_with_prompt_mask_and_no_future_result(tmp_path):
    import numpy as np
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    rows = [dict(订单号='A', 类别='办公', 金额=17)]
    items = list(sft_records(episode('repair', rows, 0, 'fixture')))
    source = tmp_path/'sft.jsonl'
    source.write_text(''.join(json.dumps(i, ensure_ascii=False)+'\n' for i in items))
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([r['prompt']+r['response'] for r in items], trainers.BpeTrainer(
        vocab_size=512, special_tokens=['<eos>', '<user>', '<assistant>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer = tmp_path/'tokenizer.json'
    tok.save(str(tokenizer))
    output = tmp_path/'prepared'
    result = subprocess.run([sys.executable, 'scripts/prepare_data.py', '--input', str(source),
        '--output', str(output), '--tokenizer', str(tokenizer), '--kind', 'sft',
        '--vocab-size', str(tok.get_vocab_size()), '--seq-len', '16384'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    ids = np.fromfile(output/'tokens.bin', dtype='<u4')
    labels = np.fromfile(output/'labels.bin', dtype='<i4')
    offsets = np.load(output/'offsets.npy')
    assert len(offsets) == len(items)+1
    for i, item in enumerate(items):
        start, end = offsets[i:i+2]
        n = len(tok.encode('<user>'+item['prompt']+'<assistant>', add_special_tokens=False).ids)
        assert (labels[start:start+n] == -100).all()
        assert (labels[start+n:end] == ids[start+n:end]).all()
        answer = tok.decode(ids[start+n:end-1].tolist(), skip_special_tokens=False)
        assert answer == item['response']
