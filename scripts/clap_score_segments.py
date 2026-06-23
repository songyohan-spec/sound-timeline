from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int = 48_000) -> tuple[np.ndarray, int]:
    import librosa

    try:
        audio, sr = sf.read(path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
    except Exception:
        mono, sr = librosa.load(path, sr=None, mono=True)
        mono = mono.astype(np.float32)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return mono, sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--prompts", type=Path, default=Path("configs/clap_prompts.json"))
    parser.add_argument("--out", type=Path, default=Path("outputs/clap_segments.json"))
    args = parser.parse_args()

    audio, sr = load_audio(args.audio)
    seg_len = int(args.segment_seconds * sr)
    hop_len = int(args.hop_seconds * sr)
    if len(audio) < seg_len:
        audio = np.pad(audio, (0, seg_len - len(audio)))

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        start = 0
        index = 0
        while start + seg_len <= len(audio):
            end = start + seg_len
            segment_path = tmp_dir / f"segment_{index:04d}.wav"
            sf.write(segment_path, audio[start:end], sr)
            segment_out = tmp_dir / f"segment_{index:04d}.json"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/clap_score_audio.py",
                    "--audio",
                    str(segment_path),
                    "--prompts",
                    str(args.prompts),
                    "--out",
                    str(segment_out),
                ],
                check=True,
            )
            report = json.loads(segment_out.read_text(encoding="utf-8"))
            segments.append(
                {
                    "index": index,
                    "start": round(start / sr, 3),
                    "end": round(end / sr, 3),
                    "scores": report["scores"],
                }
            )
            index += 1
            start += hop_len

    final = {
        "audio": str(args.audio),
        "segment_seconds": args.segment_seconds,
        "hop_seconds": args.hop_seconds,
        "segments": segments,
        "caution": "CLAP segment scores are semantic similarity hints, not ground truth labels.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()

