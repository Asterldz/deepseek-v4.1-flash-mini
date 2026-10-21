"""Download pinned public SFT files without executing dataset scripts or source code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request


def fetch(source, output):
    if not re.fullmatch(r'[a-z_]+', source['name']) or not re.fullmatch(r'[a-f0-9]{40}', source['revision']):
        raise ValueError('Require a safe local name and pinned commit')
    records = []
    for remote, local in [(source['file'], source['name']+Path(source['file']).suffix),
                          ('README.md', source['name']+'.README.md')]:
        url = 'https://huggingface.co/datasets/'+source['repo']+'/resolve/'+source['revision']+'/'+urllib.parse.quote(remote, safe='/')
        destination = output/local
        if destination.exists():
            raise ValueError(f'Refusing to replace {destination}')
        temporary = destination.with_suffix(destination.suffix+'.partial')
        digest, size = hashlib.sha256(), 0
        request = urllib.request.Request(url, headers={'User-Agent': 'v41-code-tool-data/1.0'})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, temporary.open('xb') as target:
                while chunk := response.read(1024*1024):
                    size += len(chunk)
                    if size > 600_000_000:
                        raise ValueError('Source exceeds 600MB per-file download limit')
                    digest.update(chunk)
                    target.write(chunk)
            if remote == source['file'] and (size != source['expected_bytes'] or digest.hexdigest() != source['expected_sha256']):
                raise ValueError('Download differs from pinned upstream size/SHA-256; possibly truncated')
            temporary.replace(destination)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        records.append(dict(path=local, url=url, bytes=size, sha256=digest.hexdigest()))
        print(json.dumps(dict(downloaded=local, bytes=size)), flush=True)
    result = dict(**source, files=records, downloaded_at=datetime.now(timezone.utc).isoformat(),
                full_source_downloaded=True, upstream_data_sha256_verified=True,
                semantic_quality_verified=False, source_code_executed=False)
    (output/(source['name']+'.manifest.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', default='configs/code_tool_sources.json')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = json.loads(Path(args.sources).read_text())['sources']
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda s: fetch(s, output), sources))
    manifest = dict(sources=results, status='downloaded_public_candidates_not_quality_approved',
                    source_lock_sha256=hashlib.sha256(Path(args.sources).read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
