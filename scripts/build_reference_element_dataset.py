from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def load_training_plan(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[row["canonical_label"]] = row
    return rows


def load_audio(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
    except Exception:
        import librosa

        mono, sr = librosa.load(path, sr=None, mono=True)
        mono = mono.astype(np.float32)
    if sr != target_sr:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(sr, target_sr))
        mono = resample_poly(mono, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    return mono, sr


def normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def crop_or_pad(audio: np.ndarray, target_len: int, mode: str) -> np.ndarray:
    if len(audio) >= target_len:
        if mode == "random":
            start = random.randint(0, len(audio) - target_len)
        else:
            start = max(0, (len(audio) - target_len) // 2)
        return audio[start : start + target_len]
    repeats = int(np.ceil(target_len / max(1, len(audio))))
    return np.tile(audio, repeats)[:target_len]


def write_clip(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    fade_len = min(int(sample_rate * 0.02), len(audio) // 4)
    if fade_len > 1:
        fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        audio = audio.copy()
        audio[:fade_len] *= fade
        audio[-fade_len:] *= fade[::-1]
    sf.write(path, normalize(audio), sample_rate)


def discover_sources(root: Path) -> dict[str, list[Path]]:
    discovered = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        files = sorted(path for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
        if files:
            discovered[folder.name] = files
    return discovered


def build_rows(
    source_root: Path,
    plan: dict[str, dict],
    out_dir: Path,
    duration: float,
    sample_rate: int,
    clips_per_file: int,
    crop_mode: str,
) -> tuple[list[dict], list[dict]]:
    discovered = discover_sources(source_root)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    target_len = int(duration * sample_rate)
    rows = []
    summary = []
    index = 0

    for label, files in sorted(discovered.items()):
        plan_row = plan.get(label, {})
        group = plan_row.get("group", "unknown")
        priority = plan_row.get("priority", "unknown")
        for source_file in files:
            audio, sr = load_audio(source_file, sample_rate)
            for clip_index in range(clips_per_file):
                clip = crop_or_pad(audio, target_len, crop_mode if clip_index else "center")
                out_file = audio_dir / f"ref_{index:06d}_{safe_name(label)}.wav"
                write_clip(clip, out_file, sr)
                rows.append(
                    {
                        "file": out_file.relative_to(out_dir).as_posix(),
                        "duration": duration,
                        "label": label,
                        "group": group,
                        "priority": priority,
                        "source_file": source_file.as_posix(),
                        "source_index": clip_index,
                    }
                )
                index += 1
        summary.append(
            {
                "label": label,
                "group": group,
                "priority": priority,
                "source_files": len(files),
                "clips": len(files) * clips_per_file,
                "target_min_examples": plan_row.get("target_min_examples", ""),
                "status": "ok" if files else "empty",
            }
        )
    return rows, summary


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(summary: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["label", "group", "priority", "source_files", "clips", "target_min_examples", "status"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--training-plan", type=Path, default=Path("outputs/reference_training_plan.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/reference_element_dataset"))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--clips-per-file", type=int, default=3)
    parser.add_argument("--crop-mode", choices=["center", "random"], default="random")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = load_training_plan(args.training_plan)
    rows, summary = build_rows(
        args.source_root,
        plan,
        args.out,
        args.duration,
        args.sample_rate,
        args.clips_per_file,
        args.crop_mode,
    )
    if not rows:
        raise SystemExit(
            f"No audio files found under {args.source_root}. Put WAV/FLAC/OGG/AIFF/MP3/M4A files into label folders first."
        )
    write_jsonl(rows, args.out / "metadata.jsonl")
    write_summary(summary, args.out / "summary.csv")
    print(f"wrote: {args.out / 'metadata.jsonl'}")
    print(f"wrote: {args.out / 'summary.csv'}")
    print(f"clips: {len(rows)}")


if __name__ == "__main__":
    main()
