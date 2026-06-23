from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def flatten_forbidden(row: dict) -> list[str]:
    labels = []
    for label in row.get("forbid_labels", []):
        labels.append(f"label:{label}")
    for group in row.get("forbid_groups", []):
        labels.append(f"group:{group}")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feedback-dataset", type=Path, default=Path("data/feedback_segments"))
    parser.add_argument("--out", type=Path, default=Path("models/hard_negative_suppressor.joblib"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--radius", type=float, default=2.2)
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument(
        "--feature-set",
        choices=["batch_summary", "full"],
        default="batch_summary",
        help="batch_summary uses only features available in ensemble CSV.",
    )
    args = parser.parse_args()

    rows = read_rows(args.feedback_dataset / "metadata.jsonl")
    if not rows:
        raise SystemExit("No feedback rows found.")

    examples = []
    feature_names: list[str] | None = None
    vectors = []
    for row in rows:
        stats = audio_stats(args.feedback_dataset / row["file"], quality=args.quality)
        if feature_names is None:
            if args.feature_set == "batch_summary":
                feature_names = ["centroid", "flatness", "motion_strength", "width"]
            else:
                feature_names = sorted(stats.keys())
        vector = np.array([float(stats.get(name, 0.0)) for name in feature_names], dtype=np.float32)
        vectors.append(vector)
        examples.append(
            {
                "source_file": row["source_file"],
                "start": row["start"],
                "end": row["end"],
                "forbidden": flatten_forbidden(row),
                "note": row.get("note", ""),
                "stats": stats,
            }
        )

    matrix = np.vstack(vectors)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    # With only a few feedback examples, use robust fallback scaling. Future
    # examples will make this less blunt.
    std = np.where(std < 1e-6, 1.0, std)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "examples": examples,
            "radius": args.radius,
            "strength": args.strength,
            "warning": "Memory-based hard-negative suppressor. Reliable only after multiple reviewed feedback examples.",
        },
        args.out,
    )
    print(f"examples: {len(examples)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
