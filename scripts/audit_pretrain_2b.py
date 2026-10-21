"""Independent mechanical audit for the selected 2B JSONL corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from build_pretrain_2b import RESERVED_LITERALS


def fingerprint(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit(root: Path) -> dict:
    selected = root / "selected"
    expected = json.loads((selected / "report.json").read_text())
    database = root / "audit.sqlite3"
    database.unlink(missing_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE ids (source_id TEXT PRIMARY KEY, split TEXT)")
    connection.execute("CREATE TABLE texts (hash TEXT PRIMARY KEY, split TEXT)")
    stats: dict[str, Counter] = defaultdict(Counter)
    issues = []
    required = ("source", "source_id", "group_id", "split", "revision", "license")
    for split in ("train", "validation"):
        with (selected / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                source = row.get("source", "UNKNOWN")
                counter = stats[f"{split}:{source}"]
                counter["documents"] += 1
                if row.get("split") != split:
                    counter["wrong_split"] += 1
                for field in required:
                    if not isinstance(row.get(field), str) or not row[field].strip():
                        counter[f"missing_{field}"] += 1
                source_id = row.get("source_id")
                try:
                    connection.execute("INSERT INTO ids VALUES (?, ?)", (source_id, split))
                except sqlite3.IntegrityError:
                    counter["duplicate_source_id"] += 1
                    if len(issues) < 100:
                        issues.append({"kind": "duplicate_source_id", "split": split, "line": line_number, "source_id": source_id})
                if "text" in row:
                    text = row["text"]
                    if not isinstance(text, str) or not text.strip():
                        counter["invalid_text"] += 1
                        continue
                    counter["characters"] += len(text)
                    if any(literal in text for literal in RESERVED_LITERALS):
                        counter["reserved_literal"] += 1
                    digest = fingerprint(text)
                    previous = connection.execute("SELECT split FROM texts WHERE hash=?", (digest,)).fetchone()
                    if previous:
                        counter["normalized_duplicate"] += 1
                        if previous[0] != split:
                            counter["train_validation_overlap"] += 1
                    else:
                        connection.execute("INSERT INTO texts VALUES (?, ?)", (digest, split))
                else:
                    ids = row.get("input_ids")
                    labels = row.get("labels")
                    if not isinstance(ids, list) or ids != labels or len(ids) < 2 or any(type(token) is not int or not 0 <= token < 32768 for token in ids):
                        counter["invalid_pretokenized_tail"] += 1
                    else:
                        counter["pretokenized_tail"] += 1
                if line_number % 10000 == 0:
                    connection.commit()
    connection.close()
    blockers = {}
    for key, counter in stats.items():
        for name, count in counter.items():
            if count and (name.startswith("missing_") or name in {
                "wrong_split", "duplicate_source_id", "invalid_text", "reserved_literal",
                "normalized_duplicate", "train_validation_overlap", "invalid_pretokenized_tail",
            }):
                blockers[f"{key}:{name}"] = count
    for source in ("python", "zh", "technical"):
        actual = stats[f"train:{source}"]["documents"]
        wanted = expected[source]["train_documents"]
        if actual != wanted:
            blockers[f"train:{source}:document_count_mismatch"] = {"actual": actual, "expected": wanted}
    output = {
        "status": "pass" if not blockers else "fail",
        "stats": {key: dict(value) for key, value in stats.items()},
        "blockers": blockers,
        "issues": issues,
    }
    (root / "audit_report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    database.unlink(missing_ok=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pretrain_2b")
    args = parser.parse_args()
    result = audit(Path(args.root))
    print(json.dumps({"status": result["status"], "blockers": result["blockers"]}, ensure_ascii=False, indent=2))
    if result["blockers"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
