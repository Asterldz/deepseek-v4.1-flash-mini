"""Create a small, deterministic human-review sample from the final 2B corpus."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/pretrain_2b")
    parser.add_argument("--train-per-source", type=int, default=3)
    parser.add_argument("--validation-per-source", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=1800)
    args = parser.parse_args()
    root = Path(args.root)
    rng = random.Random(20260916)
    samples: dict[str, list[dict]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    for split, limit in (("train", args.train_per_source), ("validation", args.validation_per_source)):
        path = root / "selected" / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                if "text" not in row:
                    continue
                key = f"{split}:{row['source']}"
                seen[key] += 1
                item = {
                    "split": split,
                    "source": row["source"],
                    "source_id": row["source_id"],
                    "group_id": row["group_id"],
                    "revision": row["revision"],
                    "license": row["license"],
                    "line_number": line_number,
                    "full_characters": len(row["text"]),
                    "excerpt": row["text"][: args.max_chars],
                    "excerpt_truncated": len(row["text"]) > args.max_chars,
                }
                for metadata_key in ("path", "repo", "score", "int_score", "technical_terms", "url"):
                    if metadata_key in row:
                        item[metadata_key] = row[metadata_key]
                bucket = samples[key]
                if len(bucket) < limit:
                    bucket.append(item)
                else:
                    slot = rng.randrange(seen[key])
                    if slot < limit:
                        bucket[slot] = item
    review = root / "review_samples.jsonl"
    with review.open("w", encoding="utf-8") as output:
        for key in sorted(samples):
            for item in samples[key]:
                output.write(json.dumps(item, ensure_ascii=False) + "\n")
    markdown = root / "review_samples.md"
    with markdown.open("w", encoding="utf-8") as output:
        output.write("# 2B 预训练语料随机审阅样本\n\n")
        output.write("固定随机种子 `20260916`；训练集每类 3 条，验证集每类 1 条。正文最多展示 1,800 字符。\n\n")
        for key in sorted(samples):
            output.write(f"## {key}\n\n")
            for index, item in enumerate(samples[key], 1):
                output.write(f"### 样本 {index}\n\n")
                output.write(f"- `source_id`: `{item['source_id']}`\n")
                output.write(f"- `revision`: `{item['revision']}`\n")
                output.write(f"- `license`: `{item['license']}`\n")
                if item.get("path"):
                    output.write(f"- `path`: `{item['path']}`\n")
                if item.get("technical_terms"):
                    output.write(f"- 技术词命中：`{', '.join(item['technical_terms'])}`\n")
                if item.get("int_score") is not None:
                    output.write(f"- FineWeb-Edu 质量分：`{item['int_score']}`\n")
                output.write(f"- 原文字符数：{item['full_characters']:,}\n\n")
                output.write("```text\n" + item["excerpt"].replace("```", "` ` `") + "\n```\n\n")
    print(json.dumps({"markdown": str(markdown), "jsonl": str(review), "groups": {key: len(value) for key, value in samples.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
