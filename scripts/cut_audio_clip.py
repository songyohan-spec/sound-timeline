from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int | None) -> tuple[np.ndarray, int]:
    import librosa

    try:
        audio, sr = sf.read(path, always_2d=True)
    except Exception:
        loaded, sr = librosa.load(path, sr=None, mono=False)
        if loaded.ndim == 1:
            audio = loaded[:, None]
        else:
            audio = loaded.T

    audio = audio.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr))
        min_len = min(len(ch) for ch in channels)
        audio = np.stack([ch[:min_len] for ch in channels], axis=1)
        sr = target_sr
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2], sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=44_100)
    args = parser.parse_args()

    if args.duration <= 0:
        raise SystemExit("--duration must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")

    audio, sr = load_audio(args.input, args.sample_rate)
    start_sample = int(args.start * sr)
    end_sample = start_sample + int(args.duration * sr)
    if start_sample >= len(audio):
        raise SystemExit(f"Start time {args.start}s is beyond the audio duration.")

    clip = audio[start_sample:min(end_sample, len(audio))]
    target_len = int(args.duration * sr)
    if len(clip) < target_len:
        clip = np.pad(clip, ((0, target_len - len(clip)), (0, 0)))

    fade_len = min(int(0.025 * sr), len(clip) // 4)
    if fade_len > 1:
        fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        clip[:fade_len] *= fade[:, None]
        clip[-fade_len:] *= fade[::-1, None]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, clip, sr)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

