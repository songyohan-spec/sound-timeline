from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


META_FIELDS = [
    "file",
    "labels",
    "groups",
    "primary_label",
    "primary_group",
    "source_file",
    "source_index",
    "duration",
    "training_labels",
]


def group_for_source_kind(source_kind: str) -> str:
    if "vocal" in source_kind or "rap" in source_kind:
        return "vocals"
    if any(token in source_kind for token in ["drum", "kick", "snare", "hat", "percussion", "breakbeat"]):
        return "drums"
    if any(token in source_kind for token in ["bass", "808", "sub"]):
        return "bass"
    if "guitar" in source_kind or "string" in source_kind or "violin" in source_kind:
        return "guitar_strings"
    if any(token in source_kind for token in ["noise", "fx", "riser", "impact"]):
        return "noise_fx"
    if any(token in source_kind for token in ["sample", "loop", "keys", "piano", "organ"]):
        return "sampled_loop"
    return "synth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/demucs_stems_full/weak_source_kind_harvest.csv"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--out", type=Path, default=Path("outputs/weak_source_kind_features.csv"))
    args = parser.parse_args()

    rows = []
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as file:
        manifest_rows = list(csv.DictReader(file))

    for idx, row in enumerate(manifest_rows, 1):
        audio_path = Path(row["out_file"])
        stats = audio_stats(audio_path, quality=args.quality)
        source_kind = row["source_kind"]
        out = {
            "file": audio_path.as_posix(),
            "labels": source_kind,
            "groups": group_for_source_kind(source_kind),
            "primary_label": source_kind,
            "primary_group": group_for_source_kind(source_kind),
            "source_file": row["track"],
            "source_index": f"{row['stem']}:{row['start']}-{row['end']}",
            "duration": float(row["end"]) - float(row["start"]),
            "training_labels": row["training_label"],
        }
        out.update(stats)
        rows.append(out)
        if idx % 50 == 0:
            print(f"processed {idx}/{len(manifest_rows)}")

    if not rows:
        raise SystemExit(f"No rows found in {args.manifest}")
    stat_fields = sorted(key for key in rows[0] if key not in META_FIELDS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS + stat_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
