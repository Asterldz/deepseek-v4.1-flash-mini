"""Fetch small, distributed preview samples; never use these as a production corpus.

The preview API cannot pin a Hub revision. Capture observed revisions before/after,
URLs and row coordinates for review; production downloads must use pinned shards.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import urllib.parse
import urllib.request


SOURCES = [
    ('zh_general', 'openbmb/Ultra-FineWeb', 'default', 'zh', 'content'),
    ('en_educational', 'HuggingFaceFW/fineweb-edu', 'sample-10BT', 'train', 'text'),
    ('math_web', 'open-web-math/open-web-math', 'default', 'train', 'text'),
    ('math_filtered', 'openbmb/UltraData-Math', 'UltraData-Math-L2-preview', 'train', 'content'),
]


def get_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'v41-corpus-review/1.0'})
    with urllib.request.urlopen(request, timeout=45) as stream:
        payload = stream.read(16_000_001)
    if len(payload) > 16_000_000:
        raise ValueError('Preview response exceeds 16MB cap')
    return json.loads(payload)


def fetch(spec, output):
    name, dataset, config, split, field = spec
    info_url = 'https://huggingface.co/api/datasets/'+dataset
    info = get_json(info_url)
    query = dict(dataset=dataset, config=config, split=split)
    def page(offset, length):
        url = 'https://datasets-server.huggingface.co/rows?'+urllib.parse.urlencode(dict(query, offset=offset, length=length))
        return url, get_json(url)
    _, first = page(0, 1)
    total = first['num_rows_total']
    rng = random.Random(20260915)
    # Four windows, one in each quarter; 25 rows per window. Not an IID corpus sample.
    offsets = [rng.randrange(i*total//4, max(i*total//4+1, (i+1)*total//4-25)) for i in range(4)]
    records, requests, skipped = [], [], []
    for offset in offsets:
        url, data = page(offset, 25)
        requests.append(url)
        for entry in data['rows']:
            row = entry['row']
            text = row.get(field)
            if entry.get('truncated_cells') or not isinstance(text, str) or not text.strip():
                skipped.append(entry['row_idx'])
                continue
            source_id = str(row.get('url') or row.get('id') or f'{dataset}/{config}/{split}/{entry["row_idx"]}')
            group_id = str(row.get('url') or row.get('id') or 'sha256:'+hashlib.sha256(text.encode()).hexdigest())
            records.append(dict(text=text, source=dataset, source_id=source_id,
                                revision='preview-observed:'+info['sha'],
                                license=info.get('cardData', {}).get('license', 'UNKNOWN'),
                                group_id=group_id, preview_only=True,
                                upstream_metadata={key: value for key, value in row.items() if key != field}))
    after = get_json(info_url)
    if after['sha'] != info['sha']:
        raise ValueError('Hub revision changed during preview collection')
    path = output/f'{name}.jsonl'
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False)+'\n')
    return dict(name=name, dataset=dataset, config=config, split=split, text_field=field,
                observed_hub_revision=info['sha'], revision_pinned=False, source_rows=total,
                sample_rows=len(records), offsets=offsets, skipped_truncated_or_empty=skipped,
                license_label=info.get('cardData', {}).get('license'), request_urls=requests,
                sample_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda spec: fetch(spec, output), SOURCES))
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), purpose='candidate preview review only',
                    full_corpus_quality_verified=False, sources=results)
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps([dict(name=r['name'], rows=r['sample_rows'], revision=r['observed_hub_revision']) for r in results]))


if __name__ == '__main__':
    main()
