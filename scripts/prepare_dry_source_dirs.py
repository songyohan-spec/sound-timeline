from __future__ import annotations

import argparse
from pathlib import Path


SOURCE_FAMILIES = ["synth", "bass", "guitar_like", "vocal_like", "noise_fx"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/dry_sources"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for source in SOURCE_FAMILIES:
        folder = args.root / source
        folder.mkdir(parents=True, exist_ok=True)
        readme = folder / "README.txt"
        if not readme.exists():
            readme.write_text(
                "Put dry, minimally processed WAV/FLAC/OGG/AIFF files for this source family here.\n"
                "Avoid wet reverb/delay, heavy compression, mastered loops, or mixed full-song excerpts.\n",
                encoding="utf-8",
            )

    print(f"created dry source folders under {args.root}")
    for source in SOURCE_FAMILIES:
        print(f"- {args.root / source}")


if __name__ == "__main__":
    main()
