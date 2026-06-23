from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True, help="A 3-8 second WAV/MP3/FLAC clip, or a longer file to segment.")
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/external"))
    args = parser.parse_args()

    stem = args.name or args.audio.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{stem}_segments.json"
    html_path = args.out_dir / f"{stem}_segments.html"

    run(
        [
            sys.executable,
            "scripts/infer_audio_segments.py",
            "--model",
            str(args.model),
            "--audio",
            str(args.audio),
            "--segment-seconds",
            str(args.segment_seconds),
            "--hop-seconds",
            str(args.hop_seconds),
            "--out",
            str(json_path),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/render_segments_report.py",
            "--input",
            str(json_path),
            "--out-html",
            str(html_path),
        ]
    )
    print(f"report: {html_path}")


if __name__ == "__main__":
    main()

