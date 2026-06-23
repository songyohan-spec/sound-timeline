from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def top_items(profile: dict, family_prefix: str | None = None, limit: int = 12) -> list[dict]:
    items = profile.get("ranked_palette", [])
    if family_prefix is None:
        return items[:limit]
    return [item for item in items if str(item.get("family", "")).startswith(family_prefix)][:limit]


def item_at(items: list[dict], index: int) -> dict:
    if index >= len(items):
        return {}
    return items[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-dir", type=Path, default=Path("outputs/palette_timelines_v2"))
    parser.add_argument("--out", type=Path, default=Path("outputs/palette_training_segments.csv"))
    args = parser.parse_args()

    timeline_paths = sorted(args.timeline_dir.glob("*_palette_timeline.json"))
    if not timeline_paths:
        raise SystemExit(f"No timeline JSON files found in {args.timeline_dir}")

    rows = []
    for timeline_path in timeline_paths:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        for segment in timeline.get("segments", []):
            profile = segment.get("profile", {})
            primary_sound = profile.get("primary_sound") or {}
            primary_processing = profile.get("primary_processing") or {}
            ranked = top_items(profile, limit=12)
            source_layers = profile.get("source_layers", [])
            processing_cues = profile.get("processing_cues", [])
            row = {
                "timeline": timeline_path.name,
                "segment_index": segment.get("index"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "primary_source_family": primary_sound.get("family", "none"),
                "primary_source_label": primary_sound.get("label", "none"),
                "primary_source_score": primary_sound.get("score", 0.0),
                "primary_processing_label": primary_processing.get("label", "none"),
                "primary_processing_score": primary_processing.get("score", 0.0),
                "top_source_layers": ";".join(item.get("label", "") for item in source_layers[:6]),
                "top_processing_cues": ";".join(item.get("label", "") for item in processing_cues[:6]),
            }
            for index in range(5):
                item = item_at(ranked, index)
                row[f"rank_{index + 1}_family"] = item.get("family", "")
                row[f"rank_{index + 1}_label"] = item.get("label", "")
                row[f"rank_{index + 1}_score"] = item.get("score", "")
            rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
