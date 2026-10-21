"""Generate a small exact-arithmetic supplement, not broad mathematical reasoning data."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random


def make_example(rng):
    task = rng.choice(['add', 'subtract', 'multiply', 'divide', 'linear'])
    a, b = rng.randint(-99, 99), rng.randint(-99, 99)
    if task in ('divide', 'linear'):
        a = rng.randint(1, 20)
    if task == 'add':
        a, b = sorted((a, b))
        answer = Fraction(a+b)
        question = f'计算 ({a}) + ({b})。'
        solution = f'把两个数相加，({a}) + ({b}) = {answer}。'
    elif task == 'subtract':
        answer = Fraction(a-b)
        question = f'计算 ({a}) - ({b})。'
        solution = f'减去一个数等于加上它的相反数：({a}) - ({b}) = ({a}) + ({-b}) = {answer}。'
    elif task == 'multiply':
        a, b = sorted((a, b))
        answer = Fraction(a*b)
        question = f'计算 ({a}) × ({b})。'
        solution = f'两数的乘积为 ({a}) × ({b}) = {answer}。'
    elif task == 'divide':
        answer = Fraction(b, a)
        question = f'计算 ({b}) ÷ {a}，结果用整数或最简分数表示。'
        solution = f'先写成分数 ({b})/{a}，约分后的结果为 {answer}。'
    else:
        x = rng.randint(-99, 99)
        c = a*x+b
        answer = Fraction(x)
        question = f'解方程 {a}x + ({b}) = {c}。'
        solution = f'两边减去 ({b})，得到 {a}x = {c}-({b}) = {c-b}。再除以 {a}，得到 x = {answer}。代回：{a}×({answer})+({b}) = {c}，成立。'
    operands = [a, b, c] if task == 'linear' else [a, b]
    fingerprint = hashlib.sha256(json.dumps([task, operands]).encode()).hexdigest()
    item = dict(text=f'题目：{question}\n解答：{solution}\n答案：{answer}',
                source='project/exact-arithmetic-v1', source_id=fingerprint,
                group_id='exact-arithmetic:'+fingerprint, revision='generator-v1',
                license='project-generated', task=task, operands=operands, answer=str(answer),
                verification='integer_and_fraction_arithmetic_only')
    verify_example(item)
    return item


def verify_example(item):
    a, b, *rest = item['operands']
    answer = Fraction(item['answer'])
    task = item['task']
    checks = {
        'add': lambda: answer-a == b,
        'subtract': lambda: answer+b == a,
        'multiply': lambda: answer == Fraction(a)*Fraction(b),
        'divide': lambda: a != 0 and answer*a == b,
        'linear': lambda: a != 0 and a*answer+b == rest[0],
    }
    if task not in checks or not checks[task]():
        raise ValueError('Generated arithmetic failed exact verification')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--examples', type=int, default=10000, help='Unique examples including held-out split; maximum 20000')
    p.add_argument('--seed', type=int, default=20260915)
    args = p.parse_args()
    if not 100 <= args.examples <= 20000:
        p.error('Use 100..20000 examples; this repetitive supplement should not dominate pretraining')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    rng, seen, counts = random.Random(args.seed), set(), {'train': 0, 'validation': 0}
    with (out/'train.jsonl').open('w') as train, (out/'validation.jsonl').open('w') as val:
        while len(seen) < args.examples:
            item = make_example(rng)
            key = item['source_id']
            if key in seen:
                continue
            seen.add(key)
            split = 'validation' if int(key[:8], 16) % 20 == 0 else 'train'
            stream = val if split == 'validation' else train
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
            counts[split] += 1
    meta = dict(seed=args.seed, unique_examples=len(seen), exact_verified=len(seen), splits=counts,
                generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations='Five arithmetic templates only. Held-out operand families test interpolation, not broad math reasoning. Keep a small token share; do not use as the sole final evaluation.')
    (out/'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
