from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TARGETS = [
    "source_family",
    "source_origin",
    "reverb",
    "distortion",
    "filter",
    "filter_presence",
    "filter_motion_type",
    "stereo",
    "motion",
    "motion_presence",
    "modulation",
]


def read_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/loop_synthetic"))
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")
    print(f"samples: {len(rows)}")
    for target in TARGETS:
        if target not in rows[0]:
            continue
        print(f"\n[{target}]")
        counts = Counter(row[target] for row in rows)
        for label, count in counts.most_common():
            print(f"{label:24s} {count:6d}")


if __name__ == "__main__":
    main()
