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


META_FIELDS = [
    "file",
    "label",
    "group",
    "priority",
    "source_file",
    "source_identity",
    "source_index",
    "duration",
]


def read_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_identity(source_file: str, source_index: object = "") -> str:
    if source_file.startswith("generated:"):
        return f"{source_file}:{source_index}"
    name = Path(source_file).name
    # Bootstrap files are copied into many target labels. Removing the target
    # source-family prefix helps split identical originals together later.
    for prefix in ("bootstrap_bass_", "bootstrap_synth_", "bootstrap_guitar_like_", "bootstrap_vocal_like_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def process_one(dataset: Path, row: dict, quality: str) -> dict:
    stats = audio_stats(dataset / row["file"], quality=quality)
    output = {
        "file": row["file"],
        "label": row["label"],
        "group": row["group"],
        "priority": row.get("priority", ""),
        "source_file": row.get("source_file", ""),
        "source_identity": source_identity(row.get("source_file", ""), row.get("source_index", "")),
        "source_index": row.get("source_index", ""),
        "duration": row.get("duration", ""),
    }
    output.update(stats)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/reference_element_dataset_bootstrap"))
    parser.add_argument("--out", type=Path, default=Path("outputs/reference_element_features_bootstrap.csv"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")
    if not rows:
        raise SystemExit("No metadata rows found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    features: list[dict] = []
    if args.workers <= 1:
        for idx, row in enumerate(rows, 1):
            features.append(process_one(args.dataset, row, args.quality))
            if idx % 50 == 0:
                print(f"processed {idx}/{len(rows)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_one, args.dataset, row, args.quality) for row in rows]
            for idx, future in enumerate(as_completed(futures), 1):
                features.append(future.result())
                if idx % 50 == 0:
                    print(f"processed {idx}/{len(rows)}")

    stat_fields = sorted(k for k in features[0].keys() if k not in META_FIELDS)
    with args.out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=META_FIELDS + stat_fields)
        writer.writeheader()
        writer.writerows(features)

    print(f"rows: {len(features)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
