"""Small Chinese execution-verified SFT set from fixed, authored program templates.

This is NOT a sandbox for arbitrary model/public code. Only exact programs in
PROGRAMS execute. Evaluation splits hold out input tables, not task templates.
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sqlite3


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


PROGRAMS = {
    'sum': 'result = sum(row["金额"] for row in data["rows"])',
    'filter': 'result = [row for row in data["rows"] if row["金额"] >= data["threshold"]]',
    'group': 'result = {}\nfor row in data["rows"]:\n    key = row["类别"]\n    result[key] = result.get(key, 0) + row["金额"]',
    'sort': 'result = sorted(data["rows"], key=lambda row: row["金额"], reverse=True)',
    'dedup': 'result = []\nseen = set()\nfor row in data["rows"]:\n    if row["订单号"] not in seen:\n        seen.add(row["订单号"])\n        result.append(row)',
    'mean': 'values = [row["金额"] for row in data["rows"] if row["金额"] is not None]\nresult = round(sum(values) / len(values), 2) if values else None',
    'csv_sum': 'result = sum(int(row["金额"]) for row in data["rows"])',
    'bad_column': 'result = sum(row["amount"] for row in data["rows"])',
}
SQL = 'SELECT category, SUM(amount) AS total FROM sales GROUP BY category ORDER BY category'
TOOLS = [
    dict(name='python_data', description='在预置 Python 数据处理环境中运行代码，输入为 JSON data，读取 result 变量。',
         parameters=dict(type='object', properties=dict(code=dict(type='string'), data=dict(type='object')), required=['code', 'data'])),
    dict(name='csv_read', description='解析内联 CSV 文本，返回列名和字符串类型的行数据。',
         parameters=dict(type='object', properties=dict(content=dict(type='string')), required=['content'])),
    dict(name='sqlite_select', description='对给定 sales 表运行只读查询；列为 category TEXT、amount INTEGER。',
         parameters=dict(type='object', properties=dict(query=dict(type='string'), rows=dict(type='array')), required=['query', 'rows'])),
]


def execute(name, arguments):
    """Execute only generator-authored templates; never arbitrary downloaded code."""
    if name == 'python_data':
        if set(arguments) != {'code', 'data'} or arguments['code'] not in PROGRAMS.values():
            raise ValueError('Only exact authored Python templates are allowed')
        # Round-trip ensures data contains only JSON values, never executable objects.
        data = json.loads(dumps(arguments['data']))
        if len(dumps(data)) > 20_000:
            raise ValueError('Fixture too large')
        scope = {'data': data}
        builtins = dict(sum=sum, sorted=sorted, set=set, round=round, len=len, int=int)
        try:
            exec(arguments['code'], {'__builtins__': builtins}, scope)
            return dict(ok=True, result=scope['result'])
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            return dict(ok=False, error_type=type(error).__name__, message=str(error))
    if name == 'csv_read':
        if set(arguments) != {'content'} or len(arguments['content']) > 20_000:
            raise ValueError('Invalid CSV fixture')
        reader = csv.DictReader(io.StringIO(arguments['content']))
        rows = list(reader)
        return dict(ok=True, columns=reader.fieldnames, rows=rows)
    if name == 'sqlite_select':
        if set(arguments) != {'query', 'rows'} or arguments['query'] != SQL:
            raise ValueError('Only the authored read-only SQL template is allowed')
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE TABLE sales(category TEXT, amount INTEGER)')
            db.executemany('INSERT INTO sales VALUES (?, ?)', [(r['category'], r['amount']) for r in arguments['rows']])
            cursor = db.execute(arguments['query'])
            columns = [item[0] for item in cursor.description]
            return dict(ok=True, rows=[dict(zip(columns, row)) for row in cursor.fetchall()])
    raise ValueError('Unknown tool')


def call(messages, name, arguments, trainable=True):
    call_id = 'call_'+str(sum(m['role'] == 'tool' for m in messages)+1)
    messages.append(dict(role='assistant', tool_calls=[dict(id=call_id, name=name, arguments=arguments)], trainable=trainable))
    result = execute(name, arguments)
    messages.append(dict(role='tool', tool_call_id=call_id, name=name, content=result))
    return result


def oracle(task, rows, threshold):
    if task in ('sum', 'repair', 'csv'):
        total = 0
        for row in rows:
            total += row['金额']
        return total
    if task == 'filter':
        return list(filter(lambda r: r['金额'] >= threshold, rows))
    if task in ('group', 'sql'):
        categories = sorted(set(r['类别'] for r in rows))
        totals = {key: sum(r['金额'] for r in rows if r['类别'] == key) for key in categories}
        return [dict(category=k, total=v) for k, v in totals.items()] if task == 'sql' else totals
    if task == 'sort':
        # Independent stable insertion sort oracle.
        result = []
        for row in rows:
            at = 0
            while at < len(result) and result[at]['金额'] >= row['金额']:
                at += 1
            result.insert(at, row)
        return result
    if task == 'dedup':
        return [r for i, r in enumerate(rows) if r['订单号'] not in [x['订单号'] for x in rows[:i]]]
    if task == 'mean':
        from decimal import Decimal, ROUND_HALF_EVEN
        values = [Decimal(r['金额']) for r in rows if r['金额'] is not None]
        return float((sum(values)/len(values)).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN)) if values else None
    raise ValueError(task)


def episode(task, rows, threshold, group):
    expected = oracle(task, rows, threshold)
    requests = {
        'sum': '请用 Python 求这些记录的金额总和。',
        'filter': f'请用 Python 保留金额大于等于 {threshold} 的记录，并保持原顺序。',
        'group': '请用 Python 按类别汇总金额，返回类别到总金额的映射。',
        'sort': '请用 Python 按金额降序排列；金额相同的记录保持原顺序。',
        'dedup': '请用 Python 按订单号去重，保留第一次出现的记录和顺序。',
        'mean': '请用 Python 计算非空金额的平均值，保留两位小数；全部为空时返回 null。',
        'sql': '请用 SQLite 按 category 汇总 amount，结果列为 category、total，按 category 升序。',
        'repair': '请用 Python 求金额总和；如果执行报错，请根据实际字段修复后重试。',
        'csv': '请解析下面的 CSV，再用 Python 求金额总和。',
    }
    if task == 'csv':
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=['订单号', '类别', '金额'])
        writer.writeheader(); writer.writerows(rows)
        raw_input = buffer.getvalue()
    elif task == 'sql':
        raw_input = dumps([dict(category=r['类别'], amount=r['金额']) for r in rows])
    else:
        raw_input = dumps(rows)
    messages = [dict(role='user', content=requests[task]+'\n输入数据：\n'+raw_input)]
    explanation = ''
    if task == 'csv':
        result = call(messages, 'csv_read', dict(content=raw_input))
        result = call(messages, 'python_data', dict(code=PROGRAMS['csv_sum'], data=dict(rows=result['rows'])))
    elif task == 'sql':
        result = call(messages, 'sqlite_select', dict(query=SQL, rows=json.loads(raw_input)))
    else:
        data = dict(rows=rows, **({'threshold': threshold} if task == 'filter' else {}))
        if task == 'repair':
            failure = call(messages, 'python_data', dict(code=PROGRAMS['bad_column'], data=data), trainable=False)
            assert failure['error_type'] == 'KeyError'
            explanation = '报错原因是使用了不存在的 amount 字段；已改用 金额 字段重新执行。'
        result = call(messages, 'python_data', dict(code=PROGRAMS['sum' if task == 'repair' else task], data=data))
    actual = result['rows'] if task == 'sql' else result['result']
    if actual != expected:
        raise ValueError('Execution disagrees with independent oracle')
    messages.append(dict(role='assistant', content=explanation+'执行结果：'+dumps(actual), trainable=True))
    key = hashlib.sha256((group+task).encode()).hexdigest()
    return dict(id=key, source='project/zh-local-code-tools-v1', source_id=key,
                group_id=group, revision='generator-v1', license='project-generated',
                task=task, tools=TOOLS, messages=messages, expected=expected,
                verification='authored_templates_executed_and_independent_oracle_checked',
                task_templates_held_out=False)


def verify(item):
    pending = None
    last = None
    for message in item['messages']:
        if message['role'] == 'assistant' and message.get('tool_calls'):
            if pending is not None or len(message['tool_calls']) != 1:
                raise ValueError('Invalid tool-call sequence')
            pending = message['tool_calls'][0]
        elif message['role'] == 'tool':
            if pending is None or message['tool_call_id'] != pending['id'] or message['name'] != pending['name']:
                raise ValueError('Tool response has no matching call')
            last = execute(pending['name'], pending['arguments'])
            if last != message['content']:
                raise ValueError('Recorded tool result does not match execution')
            pending = None
    if pending is not None or last is None or not last['ok']:
        raise ValueError('Incomplete or failed final trajectory')
    actual = last['rows'] if item['task'] == 'sql' else last['result']
    if actual != item['expected'] or not item['messages'][-1]['content'].endswith('执行结果：'+dumps(actual)):
        raise ValueError('Final answer disagrees with tool output')


def sft_records(item):
    """One next-assistant target per record; tool results are context only."""
    history = []
    for i, message in enumerate(item['messages']):
        clean = {k: v for k, v in message.items() if k != 'trainable'}
        if message['role'] == 'assistant' and message.get('trainable', True):
            prompt = '根据工具定义和历史消息，输出下一条 assistant 消息的 JSON 对象；工具调用后等待实际返回，不要编造结果。\n'+dumps(dict(tools=item['tools'], messages=history))
            response = dumps(clean)
            yield dict(prompt=prompt, response=response, text=prompt+'\n'+response,
                       source=item['source'], source_id=item['source_id']+':'+str(i),
                       group_id=item['group_id'], revision=item['revision'], license=item['license'],
                       task=item['task'], verification=item['verification'])
        history.append(clean)


def build(output, scenarios=200, seed=20260915):
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    episodes = {k: [] for k in ('train', 'validation', 'test')}
    tasks = ['sum', 'filter', 'group', 'sort', 'dedup', 'mean', 'sql', 'repair', 'csv']
    groups = set()
    for i in range(scenarios):
        rows = [dict(订单号=f'O{rng.randrange(1, 12):02d}', 类别=rng.choice(['办公', '食品', '日用']), 金额=rng.randrange(-20, 201)) for _ in range(rng.randrange(3, 11))]
        threshold = rng.randrange(0, 200)
        group = hashlib.sha256(dumps(rows).encode()).hexdigest()
        if group in groups:
            raise ValueError('Duplicate generated table')
        groups.add(group)
        bucket = int(group[:8], 16) % 10
        split = 'test' if bucket == 0 else 'validation' if bucket == 1 else 'train'
        for task in tasks:
            selected = [dict(r) for r in rows]
            if task == 'mean':
                for j, row in enumerate(selected):
                    if j % 3 == 0 or i % 20 == 0:
                        row['金额'] = None
            item = episode(task, selected, threshold, group)
            verify(item)
            episodes[split].append(item)
    counts, files = {}, {}
    for split, items in episodes.items():
        counts[split] = dict(episodes=len(items), sft_records=0, tasks=dict(Counter(i['task'] for i in items)))
        for suffix, iterator in [('episodes', items), ('sft', (r for item in items for r in sft_records(item)))]:
            path = output/f'{split}.{suffix}.jsonl'
            with path.open('w') as stream:
                for record in iterator:
                    stream.write(dumps(record)+'\n')
                    if suffix == 'sft':
                        counts[split]['sft_records'] += 1
            files[path.name] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(seed=seed, scenarios=scenarios, splits=counts, files=files,
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    split_rule='hold out entire input tables and all their task variants, hash modulo 10: 0=test,1=validation',
                    public_code_executed=False, formal_training_started=False,
                    limitations=['Nine authored task templates only; not broad coding or agent competence.',
                                 'Validation/test hold out tables, NOT task templates; not a final generalization benchmark.',
                                 'No clarification/no-tool trajectories yet; add before general agent SFT.',
                                 'Repair examples mask the intentionally incorrect first call from SFT targets.',
                                 'Prompt/response export works with existing SFT preparation; production tokenizer/lengths not validated.'])
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--scenarios', type=int, default=200)
    p.add_argument('--seed', type=int, default=20260915)
    args = p.parse_args()
    if not 20 <= args.scenarios <= 1000:
        p.error('Use 20..1000 tables; this finite template supplement must not dominate training')
    manifest = build(Path(args.output), args.scenarios, args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
