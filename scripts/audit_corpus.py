"""Audit UTF-8 JSONL candidates without modifying text or certifying semantic quality.

Expected records: text, source, source_id, revision, license, optional group_id.
group_id should identify the original document/repository/problem family, across variants.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
import unicodedata
from tokenizers import Tokenizer
from v41_mini.integrity import sha256_file


def text_fingerprint(text):
    # Only the fingerprint is normalized; source text, code indentation and math are untouched.
    normalized = ' '.join(unicodedata.normalize('NFKC', text).split())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def quality_flags(text):
    flags = []
    if len(text.strip()) < 100:
        flags.append('short_document_review')
    if '\x00' in text or '\ufffd' in text:
        flags.append('nul_or_replacement_character')
    if sum(ord(c) < 32 and c not in '\n\r\t' for c in text) > 0:
        flags.append('control_characters')
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 8 and 1-len(set(lines))/len(lines) > 0.3:
        flags.append('repeated_lines_review')
    if re.search(r'<(?:script|style|iframe)\b', text, re.I):
        flags.append('html_script_review')
    if re.search(r'(.)\1{39}', text):
        flags.append('repeated_character_review')
    if any(s in text for s in ('<pad>', '<bos>', '<eos>', '<draft>', '<user>', '<assistant>')):
        flags.append('reserved_token_literal_review')
    return flags


def audit(inputs, heldout, output, tokenizer_path=None, sample_per_source=30):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tok = Tokenizer.from_file(str(tokenizer_path)) if tokenizer_path else None
    rng = random.Random(20260915)
    stats, samples, sample_seen, issues = defaultdict(Counter), defaultdict(list), Counter(), []
    hashes = {}
    with tempfile.TemporaryDirectory(prefix='v41-audit-') as temp:
        db = sqlite3.connect(str(Path(temp)/'fingerprints.sqlite'))
        db.execute('CREATE TABLE texts (hash TEXT PRIMARY KEY, split TEXT, location TEXT)')
        db.execute('CREATE TABLE groups (id TEXT PRIMARY KEY, split TEXT, location TEXT)')
        # Index held-out text first, so any training match is explicitly reported as leakage.
        for split, files in [('heldout', heldout), ('train', inputs)]:
            for file in files:
                file = Path(file)
                hashes[str(file.resolve())] = sha256_file(file)
                with file.open(encoding='utf-8') as stream:
                    for line_no, line in enumerate(stream, 1):
                        location = f'{file.resolve()}:{line_no}'
                        item = json.loads(line)
                        if not isinstance(item, dict) or not isinstance(item.get('text'), str) or not item['text'].strip():
                            raise ValueError(f'Expected nonempty text at {location}')
                        text = item['text']
                        source = str(item.get('source', 'UNKNOWN'))
                        key = f'{split}:{source}'
                        counters = stats[key]
                        counters['documents'] += 1
                        counters['characters'] += len(text)
                        counters['cjk_characters'] += sum('\u4e00' <= ch <= '\u9fff' for ch in text)
                        for name in ('source', 'source_id', 'revision', 'group_id'):
                            if not isinstance(item.get(name), str) or not item[name].strip():
                                counters[f'missing_{name}'] += 1
                        license_value = item.get('license')
                        if not (isinstance(license_value, str) and license_value.strip() and license_value != 'UNKNOWN'
                                or isinstance(license_value, list) and license_value
                                and all(isinstance(v, str) and v.strip() and v != 'UNKNOWN' for v in license_value)):
                            counters['missing_license'] += 1
                        if item.get('preview_only'):
                            counters['preview_only_not_training_data'] += 1
                        flags = quality_flags(text)
                        for flag in flags:
                            counters[flag] += 1
                        if tok:
                            encoded = tok.encode(text, add_special_tokens=False)
                            # Budget includes one explicit document EOS, before chunk overlap.
                            counters['tokens_with_eos'] += len(encoded.ids)+1
                            if tok.decode(encoded.ids, skip_special_tokens=False) != text:
                                counters['tokenizer_roundtrip_mismatch'] += 1
                            if len(encoded.ids) >= len(text):
                                counters['tokens_ge_characters_review'] += 1
                        fingerprint = text_fingerprint(text)
                        old = db.execute('SELECT split, location FROM texts WHERE hash=?', (fingerprint,)).fetchone()
                        if old:
                            kind = 'heldout_exact_overlap' if old[0] != split else 'normalized_duplicate'
                            counters[kind] += 1
                            if len(issues) < 200:
                                issues.append(dict(kind=kind, location=location, previous=old[1]))
                        else:
                            db.execute('INSERT INTO texts VALUES (?,?,?)', (fingerprint, split, location))
                        group = item.get('group_id')
                        if isinstance(group, str) and group:
                            old_group = db.execute('SELECT split,location FROM groups WHERE id=?', (group,)).fetchone()
                            if old_group and old_group[0] != split:
                                counters['heldout_group_overlap'] += 1
                                if len(issues) < 200:
                                    issues.append(dict(kind='heldout_group_overlap', location=location, previous=old_group[1]))
                            elif not old_group:
                                db.execute('INSERT INTO groups VALUES (?,?,?)', (group, split, location))
                        sample_seen[key] += 1
                        entry = dict(item, audit_location=location, audit_flags=flags)
                        # Reservoir sampling per source, not the first N rows of the final corpus.
                        if len(samples[key]) < sample_per_source:
                            samples[key].append(entry)
                        else:
                            slot = rng.randrange(sample_seen[key])
                            if slot < sample_per_source:
                                samples[key][slot] = entry
                        if line_no % 10000 == 0:
                            db.commit()
        db.close()
    totals = Counter()
    for values in stats.values():
        totals.update(values)
    blockers = {name: count for name, count in totals.items() if count and (
        name.startswith('missing_') or name in ('heldout_exact_overlap', 'heldout_group_overlap',
        'normalized_duplicate', 'nul_or_replacement_character', 'control_characters',
        'tokenizer_roundtrip_mismatch', 'reserved_token_literal_review', 'preview_only_not_training_data'))}
    if not inputs or not heldout:
        blockers['missing_train_or_heldout_split'] = 1
    report = dict(status='mechanical_fail' if blockers else 'mechanical_checks_passed_review_required',
                  ready_for_training=False, inputs_sha256=hashes,
                  tokenizer_sha256=sha256_file(tokenizer_path) if tok else None,
                  by_source=dict(stats), blockers=blockers, issue_examples=issues,
                  limitations=['No semantic correctness or mathematical answer verification.',
                               'Normalized exact dedup only; near-duplicate and benchmark-substring checks still required.',
                               'group_id metadata is checked for overlap but its provenance is not independently verified.',
                               'Source-level score and heuristic flags do not establish document quality.',
                               'No GPU performance/stability evidence or full-corpus representativeness is inferred.'])
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    with (output/'review_samples.jsonl').open('w', encoding='utf-8') as stream:
        for key in sorted(samples):
            for item in samples[key]:
                stream.write(json.dumps(dict(item, audit_source_group=key), ensure_ascii=False)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--heldout', nargs='*', default=[])
    p.add_argument('--output', required=True, help='New audit directory')
    p.add_argument('--tokenizer')
    p.add_argument('--sample-per-source', type=int, default=30)
    args = p.parse_args()
    if args.sample_per_source < 1:
        p.error('sample-per-source must be positive')
    report = audit(args.input, args.heldout, args.output, args.tokenizer, args.sample_per_source)
    print(json.dumps(dict(status=report['status'], blockers=report['blockers'], output=args.output), ensure_ascii=False))
    if report['blockers']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
