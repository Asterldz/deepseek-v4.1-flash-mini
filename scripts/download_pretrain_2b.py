"""Download the pinned public shards and write a local SHA-256 manifest."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


REMOTE_LAYOUT = {
    "python": ("codeparrot", ""),
    "zh": ("ultrafineweb_zh", "data/ultrafineweb_zh/"),
    "technical": ("fineweb_edu", "sample/10BT/"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(path: Path) -> None:
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rb") as handle:
            while handle.read(8 << 20):
                pass
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq
        metadata = pq.ParquetFile(path).metadata
        if metadata.num_rows < 1:
            raise ValueError(f"empty parquet shard: {path}")


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pretrain_2b.json")
    parser.add_argument("--root", default="data/pretrain_2b")
    parser.add_argument("--manifest-only", action="store_true", help="hash existing files without network access")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(args.root)
    manifest = {"config_sha256": sha256(config_path), "files": []}
    for source, spec in config["sources"].items():
        directory, prefix = REMOTE_LAYOUT[source]
        out_dir = root / "raw" / directory
        out_dir.mkdir(parents=True, exist_ok=True)
        for filename in spec["files"]:
            target = out_dir / filename
            url = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{prefix}{filename}?download=true"
            if target.exists():
                try:
                    validate(target)
                except Exception:
                    if args.manifest_only:
                        raise
                    pass
            if not target.exists():
                if args.manifest_only:
                    raise FileNotFoundError(target)
            try:
                validate(target)
            except Exception:
                wget = shutil.which("wget")
                if wget:
                    subprocess.run([
                        wget, "-q", "-c", "--tries=20", "--timeout=30",
                        "--retry-connrefused", url, "-O", str(target),
                    ], check=True)
                else:
                    cached = hf_hub_download(
                        repo_id=spec["dataset"],
                        filename=prefix + filename,
                        repo_type="dataset",
                        revision=spec["revision"],
                    )
                    partial = target.with_suffix(target.suffix + ".part")
                    shutil.copyfile(cached, partial)
                    partial.replace(target)
                validate(target)
            manifest["files"].append({
                "source": source,
                "dataset": spec["dataset"],
                "revision": spec["revision"],
                "filename": filename,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
                "url": url,
            })
    destination = root / "raw_manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
