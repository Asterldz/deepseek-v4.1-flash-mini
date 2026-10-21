"""Conservative public SFT candidate filtering; never executes public programs.

Syntax/schema checks are not semantic or execution verification. Exact first-user
hashes keep repeated tasks together across sources; paraphrase leakage remains.
"""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import warnings

from scripts.make_zh_code_tools import dumps, sft_records


DOMAIN = re.compile(r'\b(csv|json|pandas|dataframe|sqlite|sql|dataset|spreadsheet|filter|sorting|sort|aggregate|average|mean|median|duplicates?|deduplication)\b', re.I)
FENCE = re.compile(r'```(?:python|py)\s*\n(.*?)```', re.S | re.I)


def fingerprint(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).split()).encode()).hexdigest()


def python_ok(text):
    codes = FENCE.findall(text)
    if not codes:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            for code in codes:
                ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    return True


def normalize_hermes(row):
    tools = row.get('tools')
    if isinstance(tools, str):
        tools = json.loads(tools)
    messages, pending = [], []
    calls = 0
    for turn in row['conversations']:
        role, value = turn['from'], turn['value']
        if role == 'system':
            if tools is None:
                match = re.search(r'<tools>\s*(.*?)\s*</tools>', value, re.S)
                tools = json.loads(match[1]) if match else None
            continue
        if role in ('human', 'user'):
            if pending:
                raise ValueError('missing_tool_response')
            messages.append(dict(role='user', content=value))
        elif role in ('gpt', 'assistant'):
            if pending:
                raise ValueError('missing_tool_response')
            matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', value, re.S)
            if matches:
                functions = {t['function']['name']: t['function'] for t in tools}
                actions = []
                for raw in matches:
                    action = json.loads(raw)
                    name, arguments = action['name'], action['arguments']
                    if name not in functions or not isinstance(arguments, dict):
                        raise ValueError('invalid_tool')
                    schema = functions[name]['parameters']
                    if not set(schema.get('required', [])) <= arguments.keys():
                        raise ValueError('missing_required_argument')
                    properties = schema.get('properties', {})
                    if not arguments.keys() <= properties.keys():
                        raise ValueError('unknown_argument')
                    for key, val in arguments.items():
                        spec = properties[key]
                        types = dict(string=str, integer=int, number=(int, float), boolean=bool, array=list, object=dict)
                        kind = spec.get('type')
                        if kind in types and (not isinstance(val, types[kind]) or kind in ('integer', 'number') and isinstance(val, bool)):
                            raise ValueError('wrong_argument_type')
                        if 'enum' in spec and val not in spec['enum']:
                            raise ValueError('invalid_enum')
                    calls += 1
                    actions.append(dict(id=f'call_{calls}', name=name, arguments=arguments))
                pending = actions.copy()
                text = re.sub(r'<tool_call>.*?</tool_call>', '', value, flags=re.S).strip()
                messages.append(dict(role='assistant', tool_calls=actions, **({'content': text} if text else {})))
            else:
                messages.append(dict(role='assistant', content=value))
        elif role == 'tool':
            for raw in re.findall(r'<tool_response>\s*(.*?)\s*</tool_response>', value, re.S):
                response = json.loads(raw)
                if not pending or response['name'] != pending[0]['name']:
                    raise ValueError('mismatched_response')
                action = pending.pop(0)
                messages.append(dict(role='tool', tool_call_id=action['id'], name=action['name'], content=response['content']))
        else:
            raise ValueError('unknown_role')
    if not tools or pending or not 1 <= calls <= 3 or not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('incomplete_or_long_tool_trajectory')
    return tools, messages


def convert(row, name):
    if name == 'magicoder':
        if row.get('lang') != 'python':
            raise ValueError('not_python')
        question, answer = row['problem'], row['solution']
        if not DOMAIN.search(question):
            raise ValueError('outside_data_processing_keywords')
        if not python_ok(answer):
            raise ValueError('no_parseable_python')
        tools, messages = [], [dict(role='user', content=question), dict(role='assistant', content=answer)]
    elif name == 'code_feedback':
        messages = row['messages']
        tools = []
        if not messages or messages[0]['role'] != 'user' or messages[-1]['role'] != 'assistant':
            raise ValueError('invalid_conversation')
        if any(m['role'] not in ('user', 'assistant') for m in messages):
            raise ValueError('unsupported_role')
        question = messages[0]['content']
        if not DOMAIN.search(question):
            raise ValueError('outside_data_processing_keywords')
        answers = [m['content'] for m in messages if m['role'] == 'assistant']
        if not any(python_ok(a) for a in answers) or any('```python' in a and not python_ok(a) for a in answers):
            raise ValueError('no_parseable_python')
        # Preserve upstream feedback as user turns; do not falsely label it a local tool result.
    elif name == 'hermes':
        tools, messages = normalize_hermes(row)
        question = next(m['content'] for m in messages if m['role'] == 'user')
        if not DOMAIN.search(question+' '+dumps(tools)):
            raise ValueError('outside_data_processing_keywords')
    else:
        raise ValueError('unknown_source')
    serialized = dumps(dict(tools=tools, messages=messages))
    if len(serialized) > 10_000 or len(messages) > 10:
        raise ValueError('long_record_chars_not_tokens')
    if '\ufffd' in serialized or any(t in serialized for t in ('<eos>', '<user>', '<assistant>')):
        raise ValueError('encoding_or_reserved_boundary')
    group = fingerprint(question)
    return dict(tools=tools, messages=messages, group_id='public-task:'+group,
                task='public_'+name, verification='syntax_or_partial_schema_only_NOT_execution_verified',
                quality_status='candidate_requires_semantic_review', upstream_record_id=row.get('id', row.get('index')))


def curate(raw, output, limit=2500):
    manifest = json.loads((raw/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=False)
    all_items, stats, seen = [], {}, set()
    for source in manifest['sources']:
        name = source['name']
        info = source['files'][0]
        path = raw/info['path']
        digest = hashlib.sha256()
        with path.open('rb') as f:
            while chunk := f.read(1024*1024):
                digest.update(chunk)
        if digest.hexdigest() != info['sha256']:
            raise ValueError('Raw source hash mismatch')
        counts = Counter()
        selected = []
        with path.open() as stream:
            records = json.load(stream) if path.suffix == '.json' else (json.loads(line) for line in stream)
            for index, row in enumerate(records):
                counts['scanned'] += 1
                try:
                    item = convert(row, name)
                except (ValueError, KeyError, TypeError, StopIteration) as error:
                    counts['reject:'+str(error)[:100]] += 1
                    continue
                identity = fingerprint(dumps(item['messages']))
                if identity in seen:
                    counts['exact_duplicate'] += 1
                    continue
                seen.add(identity)
                selected.append(dict(item, id=identity, source_id=identity, source=source['repo'],
                                     revision=source['revision'], license=source['license_label'], upstream_row=index))
        counts['mechanically_eligible'] = len(selected)
        # Hash-based selection spans the complete source instead of selecting its first rows.
        selected.sort(key=lambda r: r['id'])
        selected = selected[:limit]
        counts['selected'] = len(selected)
        stats[name] = dict(counts, full_source_downloaded=source.get('full_source_downloaded', False),
                           upstream_data_sha256_verified=source.get('upstream_data_sha256_verified', False))
        all_items.extend(selected)
    splits, hashes, sft_seen = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        items = []
        for item in all_items:
            bucket = int(item['group_id'].split(':')[-1][:8], 16) % 10
            destination = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
            if destination == split:
                items.append(item)
        counts = dict(episodes=len(items), sft_records=0, duplicate_sft_records_removed=0,
                      by_source=dict(Counter(x['source'] for x in items)))
        for suffix, records in [('episodes', items), ('sft', (r for item in items for r in sft_records(item)))]:
            path = output/f'{split}.{suffix}.jsonl'
            with path.open('w') as stream:
                for row in records:
                    if suffix == 'sft':
                        key = fingerprint(row['text'])
                        if key in sft_seen:
                            if sft_seen[key] != split:
                                raise ValueError('Identical SFT example crosses split boundaries')
                            counts['duplicate_sft_records_removed'] += 1
                            continue
                        sft_seen[key] = split
                    row['quality_status'] = 'candidate_requires_semantic_review'
                    stream.write(dumps(row)+'\n')
                    if suffix == 'sft': counts['sft_records'] += 1
            hashes[path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        splits[split] = counts
    report = dict(sources=stats, splits=splits, files=hashes,
                  source_manifest_sha256=hashlib.sha256((raw/'manifest.json').read_bytes()).hexdigest(),
                  filter_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  ready_for_training=False, public_code_executed=False,
                  limitations=['Keyword, Python AST, partial tool schema and length filtering only; NOT semantic verification.',
                               'Character limit is not a tokenizer limit. Actual 2048-token fit must be measured later.',
                               'Most public candidates are English and not yet translated.',
                               'Exact initial-query grouping only; repository families, paraphrases and benchmark contamination not checked.',
                               'Code-Feedback roles are preserved, not relabeled as locally observed execution.',
                               'Do not concatenate with verified Chinese examples as if both have the same validation status.'])
    (output/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit-per-source', type=int, default=2500)
    args = p.parse_args()
    print(json.dumps(curate(Path(args.raw), Path(args.output), args.limit_per_source), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
