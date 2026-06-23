from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


DEFAULT_MAP = {
    "bass": [
        "sub_bass",
        "distorted_808_bass",
        "pulsing_sidechain_bass",
    ],
    "synth": [
        "lush_synth_pad",
        "syrupy_video_game_synth_melody",
        "rage_synth_lead",
        "bitcrushed_synth_lead",
        "noisy_wavetable_texture",
        "fuzzy_diy_synth_texture",
    ],
    "guitar_like": [
        "filtered_guitar_loop",
        "washed_chorus_guitar",
        "distorted_guitar_texture",
    ],
    "vocal_like": [
        "processed_lead_vocal",
        "hard_tuned_vocal",
        "pitched_vocal_chop",
        "breathy_vocal_pad",
        "stacked_harmony_vocal",
        "vocal_synth_hybrid",
    ],
    "noise_fx": [
        "digital_glitch",
        "granular_texture",
        "glitch_percussion",
        "industrial_noise_layer",
        "watery_background_texture",
    ],
}


def copy_examples(dry_root: Path, ref_root: Path, max_per_label: int) -> list[dict]:
    rows = []
    for dry_label, target_labels in DEFAULT_MAP.items():
        dry_dir = dry_root / dry_label
        if not dry_dir.exists():
            continue
        files = sorted(path for path in dry_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
        if not files:
            continue
        for target in target_labels:
            target_dir = ref_root / target
            target_dir.mkdir(parents=True, exist_ok=True)
            existing_names = {path.name for path in target_dir.iterdir() if path.is_file()}
            copied = 0
            for source in files[:max_per_label]:
                out = target_dir / f"bootstrap_{dry_label}_{source.name}"
                if out.name not in existing_names:
                    shutil.copy2(source, out)
                    copied += 1
            rows.append(
                {
                    "dry_label": dry_label,
                    "target_label": target,
                    "available": len(files),
                    "copied": copied,
                    "target_dir": target_dir.as_posix(),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-root", type=Path, default=Path("data/dry_sources"))
    parser.add_argument("--reference-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--max-per-label", type=int, default=20)
    args = parser.parse_args()

    rows = copy_examples(args.dry_root, args.reference_root, args.max_per_label)
    for row in rows:
        print(f"{row['target_label']}: copied {row['copied']} from {row['dry_label']} ({row['available']} available)")
    print(f"labels touched: {len(rows)}")


if __name__ == "__main__":
    main()
