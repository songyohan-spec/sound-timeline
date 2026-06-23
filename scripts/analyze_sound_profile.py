from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sound_profiles"))
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    args = parser.parse_args()

    name = args.name or safe_stem(args.audio)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    segments_json = args.out_dir / f"{name}_segments.json"
    palette_json = args.out_dir / f"{name}_palette.json"
    profile_json = args.out_dir / f"{name}_sound_profile.json"

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
            str(segments_json),
        ]
    )
    run([sys.executable, "scripts/clap_palette_score.py", "--audio", str(args.audio), "--out", str(palette_json)])
    run(
        [
            sys.executable,
            "scripts/create_sound_profile_json.py",
            "--palette",
            str(palette_json),
            "--segments",
            str(segments_json),
            "--title",
            name,
            "--out",
            str(profile_json),
        ]
    )
    print(f"profile: {profile_json}")


if __name__ == "__main__":
    main()

