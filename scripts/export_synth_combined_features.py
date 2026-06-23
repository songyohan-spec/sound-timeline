from __future__ import annotations

import argparse
import csv
from pathlib import Path

from export_synth_specialist_features import META_FIELDS, process_one, read_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/synth_specialist_features_v2.csv"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    args = parser.parse_args()

    features = []
    for dataset in args.datasets:
        rows = read_rows(dataset / "metadata.jsonl")
        print(f"{dataset}: {len(rows)} rows")
        for idx, row in enumerate(rows, 1):
            out = process_one(dataset, row, args.quality)
            out["dataset"] = dataset.as_posix()
            features.append(out)
            if idx % 100 == 0:
                print(f"  processed {idx}/{len(rows)}")

    if not features:
        raise SystemExit("No features produced.")
    meta_fields = META_FIELDS + ["dataset"]
    stat_fields = sorted(key for key in features[0] if key not in meta_fields)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=meta_fields + stat_fields)
        writer.writeheader()
        writer.writerows(features)
    print(f"rows: {len(features)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
