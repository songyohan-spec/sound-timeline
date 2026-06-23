from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


META_COLUMNS = {
    "file",
    "labels",
    "groups",
    "primary_label",
    "primary_group",
    "source_file",
    "source_index",
    "duration",
}


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def reverse_mapping(mapping: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for source_kind, training_labels in mapping.items():
        for label in training_labels:
            out.setdefault(label, []).append(source_kind)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/reference_mixture_features_v5_vocal.csv"))
    parser.add_argument("--map", type=Path, default=Path("configs/source_kind_training_map.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/source_kind_features_v1.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    by_training_label = reverse_mapping(mapping)

    rows = []
    dropped = 0
    for _, row in frame.iterrows():
        training_labels = split_pipe(row.get("labels", ""))
        source_kinds = sorted({kind for label in training_labels for kind in by_training_label.get(label, [])})
        if not source_kinds:
            dropped += 1
            continue
        out = row.to_dict()
        out["training_labels"] = "|".join(training_labels)
        out["labels"] = "|".join(source_kinds)
        out["groups"] = "|".join(sorted({kind.split("_", 1)[0] for kind in source_kinds}))
        out["primary_label"] = source_kinds[0]
        out["primary_group"] = out["groups"].split("|", 1)[0]
        rows.append(out)

    if not rows:
        raise SystemExit("No source-kind rows produced. Check mapping and feature labels.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"dropped: {dropped}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
