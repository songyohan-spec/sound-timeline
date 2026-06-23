from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


META_FIELDS = ["file", "label", "family", "base_label", "duration", "source_index"]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def process_one(dataset: Path, row: dict, quality: str) -> dict:
    stats = audio_stats(dataset / row["file"], quality=quality)
    out = {field: row.get(field, "") for field in META_FIELDS}
    out.update(stats)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synth_specialist_v1"))
    parser.add_argument("--out", type=Path, default=Path("outputs/synth_specialist_features_v1.csv"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")
    features = []
    if args.workers <= 1:
        for idx, row in enumerate(rows, 1):
            features.append(process_one(args.dataset, row, args.quality))
            if idx % 100 == 0:
                print(f"processed {idx}/{len(rows)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, args.dataset, row, args.quality) for row in rows]
            for idx, future in enumerate(as_completed(futures), 1):
                features.append(future.result())
                if idx % 100 == 0:
                    print(f"processed {idx}/{len(rows)}")

    stat_fields = sorted(key for key in features[0] if key not in META_FIELDS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS + stat_fields)
        writer.writeheader()
        writer.writerows(features)
    print(f"rows: {len(features)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
