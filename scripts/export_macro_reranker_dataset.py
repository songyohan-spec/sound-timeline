from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def matching_calibrated_path(calibrated_dir: Path, palette_path: Path) -> Path:
    name = palette_path.name.replace("_palette_timeline.json", "_calibrated_timeline.json")
    return calibrated_dir / name


def score_features(profile: dict) -> dict[str, float]:
    features: dict[str, float] = {}
    for item in profile.get("ranked_palette", []):
        family = item.get("family", "unknown")
        label = item.get("label", "unknown")
        key = f"score::{family}::{label}"
        features[key] = float(item.get("score") or 0.0)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette-dir", type=Path, default=Path("outputs/palette_timelines_v3"))
    parser.add_argument("--calibrated-dir", type=Path, default=Path("outputs/calibrated_timelines_v3"))
    parser.add_argument("--out", type=Path, default=Path("outputs/macro_reranker_dataset.csv"))
    args = parser.parse_args()

    rows: list[dict] = []
    feature_names: set[str] = set()
    for palette_path in sorted(args.palette_dir.glob("*_palette_timeline.json")):
        calibrated_path = matching_calibrated_path(args.calibrated_dir, palette_path)
        if not calibrated_path.exists():
            raise SystemExit(f"Missing calibrated timeline for {palette_path.name}: {calibrated_path}")

        palette_timeline = json.loads(palette_path.read_text(encoding="utf-8"))
        calibrated_timeline = json.loads(calibrated_path.read_text(encoding="utf-8"))
        calibrated_by_index = {
            segment["index"]: segment
            for segment in calibrated_timeline.get("segments", [])
        }
        for segment in palette_timeline.get("segments", []):
            index = segment.get("index")
            calibrated = calibrated_by_index.get(index)
            if calibrated is None:
                raise SystemExit(f"Missing calibrated segment {index} in {calibrated_path}")
            profile = segment.get("profile", {})
            primary_sound = profile.get("primary_sound") or {}
            primary_processing = profile.get("primary_processing") or {}
            features = score_features(profile)
            feature_names.update(features)
            row = {
                "timeline": palette_path.name,
                "segment_index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "raw_primary_source_family": primary_sound.get("family", "none"),
                "raw_primary_source_label": primary_sound.get("label", "none"),
                "raw_primary_processing_label": primary_processing.get("label", "none"),
                "target_source_macro": calibrated.get("calibrated_source", "none"),
                "target_processing_macro": calibrated.get("calibrated_processing", "none"),
                **features,
            }
            rows.append(row)

    if not rows:
        raise SystemExit("No rows exported.")

    fixed_fields = [
        "timeline",
        "segment_index",
        "start",
        "end",
        "raw_primary_source_family",
        "raw_primary_source_label",
        "raw_primary_processing_label",
        "target_source_macro",
        "target_processing_macro",
    ]
    fields = fixed_fields + sorted(feature_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, 0.0) for field in fields})
    print(f"wrote: {args.out}")
    print(f"rows: {len(rows)}")
    print(f"features: {len(feature_names)}")


if __name__ == "__main__":
    main()
