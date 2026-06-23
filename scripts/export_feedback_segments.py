from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
        return audio.astype(np.float32), sr
    except Exception:
        import librosa

        loaded, sr = librosa.load(path, sr=None, mono=False)
        if loaded.ndim == 1:
            audio = np.stack([loaded, loaded], axis=1)
        else:
            audio = loaded.T
        return audio.astype(np.float32), sr


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-root", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--feedback", type=Path, default=Path("configs/segment_feedback.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/feedback_segments"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(exist_ok=True)
    rows = []
    with args.feedback.open("r", encoding="utf-8-sig", newline="") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            path = args.audio_root / row["file"]
            audio, sr = load_audio(path)
            start = float(row["start"])
            end = float(row["end"])
            start_i = max(0, int(start * sr))
            end_i = min(len(audio), int(end * sr))
            clip = audio[start_i:end_i]
            out_file = audio_dir / f"feedback_{idx:04d}_{Path(row['file']).stem}_{start:.2f}_{end:.2f}.wav"
            sf.write(out_file, clip, sr)
            rows.append(
                {
                    "file": out_file.relative_to(args.out).as_posix(),
                    "source_file": row["file"],
                    "start": start,
                    "end": end,
                    "forbid_groups": split_pipe(row.get("forbid_groups", "")),
                    "forbid_labels": split_pipe(row.get("forbid_labels", "")),
                    "force_labels": split_pipe(row.get("force_labels", "")),
                    "note": row.get("note", ""),
                }
            )
    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"clips: {len(rows)}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
