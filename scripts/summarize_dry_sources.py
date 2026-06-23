from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_FAMILIES = ["synth", "bass", "guitar_like", "vocal_like", "noise_fx"]
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/dry_sources"))
    args = parser.parse_args()

    total = 0
    for source in SOURCE_FAMILIES:
        folder = args.root / source
        files = [path for path in folder.glob("*") if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
        total += len(files)
        print(f"{source:12s} {len(files):6d}")
    print(f"{'total':12s} {total:6d}")


if __name__ == "__main__":
    main()
