from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


DEFAULT_TARGETS = [
    "electronic_drum_machine",
    "glitch_percussion",
    "sampled_loop_texture",
    "piano_or_keyboard_loop",
    "warm_keys_or_organ",
]


def parse_score(top_source_kinds: str, label: str) -> tuple[float, float] | None:
    for part in str(top_source_kinds or "").split(";"):
        part = part.strip()
        if not part.startswith(f"{label}:"):
            continue
        try:
            score_part = part.split(":", 1)[1].split("/", 2)
            return float(score_part[0]), float(score_part[1])
        except (IndexError, ValueError):
            return None
    return None


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), sr


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def write_clip(stem_path: Path, start: float, end: float, out_path: Path) -> None:
    audio, sr = load_audio(stem_path)
    start_i = max(0, int(start * sr))
    end_i = min(len(audio), max(start_i + 1, int(end * sr)))
    clip = normalize(audio[start_i:end_i])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, clip, sr)


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem-csv", type=Path, default=Path("outputs/demucs_stems_full/stem_source_kind_strict.csv"))
    parser.add_argument("--map", type=Path, default=Path("configs/source_kind_training_map.json"))
    parser.add_argument("--source-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--max-per-kind", type=int, default=48)
    parser.add_argument("--min-score-ratio", type=float, default=1.05)
    parser.add_argument("--out-manifest", type=Path, default=Path("outputs/demucs_stems_full/weak_source_kind_harvest.csv"))
    args = parser.parse_args()

    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    targets = [item.strip() for item in args.targets.split(",") if item.strip()]
    written_counts = {target: 0 for target in targets}
    manifest_rows = []

    with args.stem_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    candidates = []
    for row in rows:
        labels = [label for label in row.get("detected_source_kinds", "").split("|") if label]
        for source_kind in labels:
            if source_kind not in written_counts:
                continue
            parsed = parse_score(row.get("top_source_kinds", ""), source_kind)
            if not parsed:
                continue
            score, threshold = parsed
            if threshold <= 0 or score / threshold < args.min_score_ratio:
                continue
            candidates.append((score / threshold, source_kind, row))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for ratio, source_kind, row in candidates:
        if written_counts[source_kind] >= args.max_per_kind:
            continue
        training_labels = mapping.get(source_kind, [])
        if not training_labels:
            continue
        training_label = training_labels[0]
        index = written_counts[source_kind]
        track = safe_name(row["track"])
        stem = safe_name(row["stem"])
        start = float(row["start"])
        end = float(row["end"])
        out_path = args.source_root / training_label / "_weak_demucs" / f"{track}_{stem}_{start:.2f}_{end:.2f}_{safe_name(source_kind)}_{index:03d}.wav"
        if not out_path.exists():
            write_clip(Path(row["stem_path"]), start, end, out_path)
        written_counts[source_kind] += 1
        manifest_rows.append(
            {
                "source_kind": source_kind,
                "training_label": training_label,
                "track": row["track"],
                "stem": row["stem"],
                "start": row["start"],
                "end": row["end"],
                "score_ratio": round(ratio, 4),
                "out_file": out_path.as_posix(),
            }
        )

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["source_kind", "training_label", "track", "stem", "start", "end", "score_ratio", "out_file"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("harvested:")
    for source_kind, count in written_counts.items():
        print(f"  {source_kind}: {count}")
    print(f"wrote: {args.out_manifest}")


if __name__ == "__main__":
    main()
