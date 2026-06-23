from __future__ import annotations

import argparse
import csv
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp_palette_score import audio_stats


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), sr


def process_file(task: tuple[str, float, float, str]) -> list[dict]:
    audio_path = Path(task[0])
    segment_seconds = task[1]
    hop_seconds = task[2]
    quality = task[3]
    audio, sr = load_audio(audio_path)
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(audio) < seg_len:
        audio = np.pad(audio, ((0, seg_len - len(audio)), (0, 0)))

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        start = 0
        index = 0
        while start + seg_len <= len(audio):
            end = start + seg_len
            segment_path = tmp_dir / f"segment_{index:04d}.wav"
            sf.write(segment_path, audio[start:end], sr)
            stats = audio_stats(segment_path, quality=quality)
            row = {
                "file": audio_path.name,
                "stem": safe_stem(audio_path),
                "segment_index": index,
                "start": round(start / sr, 3),
                "end": round(end / sr, 3),
                **stats,
            }
            rows.append(row)
            index += 1
            start += hop_len
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out", type=Path, default=Path("outputs/dsp_segment_features.csv"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--quality", choices=["librosa", "fast"], default="librosa")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    tasks = [(str(path), args.segment_seconds, args.hop_seconds, args.quality) for path in files]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_file, task) for task in tasks]
        for future in as_completed(futures):
            file_rows = future.result()
            rows.extend(file_rows)
            if file_rows:
                print(f"features: {file_rows[0]['file']} ({len(file_rows)} segments)")

    rows = sorted(rows, key=lambda row: (row["stem"], int(row["segment_index"])))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out}")
    print(f"rows: {len(rows)}")


if __name__ == "__main__":
    main()
