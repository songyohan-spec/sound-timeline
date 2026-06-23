from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_demucs_stems_source_kind import DEFAULT_STEM_ORDER, discover_stems
from dsp_palette_score import audio_stats
from infer_reference_elements_timeline import load_audio


META_FIELDS = ["track", "stem", "start", "end", "stem_path", "duration"]


def segment_stem(stem_path: Path, track: str, segment_seconds: float, hop_seconds: float, quality: str) -> list[dict]:
    audio, sr = load_audio(stem_path, sample_rate=22050)
    segment_len = max(1, int(segment_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    duration = audio.shape[0] / sr
    starts = list(range(0, max(1, audio.shape[0] - segment_len + 1), hop_len)) or [0]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, start_i in enumerate(starts):
            end_i = min(audio.shape[0], start_i + segment_len)
            clip = audio[start_i:end_i]
            if clip.shape[0] < int(0.25 * sr):
                continue
            if clip.shape[0] < segment_len:
                pad = np.zeros((segment_len - clip.shape[0], clip.shape[1]), dtype=np.float32)
                clip = np.vstack([clip, pad])
            clip_path = tmp_dir / f"{track}_{stem_path.stem}_{index:04d}.wav"
            sf.write(clip_path, clip, sr)
            stats = audio_stats(clip_path, quality=quality)
            row = {
                "track": track,
                "stem": stem_path.stem,
                "start": round(start_i / sr, 4),
                "end": round(min(duration, (start_i + segment_len) / sr), 4),
                "stem_path": stem_path.as_posix(),
                "duration": round(duration, 4),
            }
            row.update(stats)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_6s_full/htdemucs_6s"))
    parser.add_argument("--stems", default="")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_feature_cache.csv"))
    args = parser.parse_args()

    stems = discover_stems(args.stems_root, args.stems)
    stem_order = [stem for stem in DEFAULT_STEM_ORDER if stem in stems] + sorted(set(stems) - set(DEFAULT_STEM_ORDER))
    jobs = []
    for track_dir in sorted(path for path in args.stems_root.iterdir() if path.is_dir()):
        for stem in stem_order:
            stem_path = track_dir / f"{stem}.wav"
            if stem_path.exists():
                jobs.append((stem_path, track_dir.name, args.segment_seconds, args.hop_seconds, args.quality))

    if not jobs:
        raise SystemExit(f"No stem wav files found under {args.stems_root}")

    rows = []
    if args.workers <= 1:
        for idx, job in enumerate(jobs, 1):
            rows.extend(segment_stem(*job))
            if idx % 20 == 0:
                print(f"processed stems {idx}/{len(jobs)}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(segment_stem, *job) for job in jobs]
            for idx, future in enumerate(as_completed(futures), 1):
                rows.extend(future.result())
                if idx % 20 == 0:
                    print(f"processed stems {idx}/{len(jobs)}")

    if not rows:
        raise SystemExit("No cache rows produced.")
    rows.sort(key=lambda row: (row["track"], row["stem"], float(row["start"])))
    stat_fields = sorted(key for key in rows[0] if key not in META_FIELDS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS + stat_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"stems: {len(jobs)}")
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
