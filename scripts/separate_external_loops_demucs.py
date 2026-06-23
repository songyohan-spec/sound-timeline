from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def expected_stems(model: str, two_stems: str | None = None) -> tuple[str, ...]:
    if two_stems:
        return (f"{two_stems}.wav", f"no_{two_stems}.wav")
    if model == "htdemucs_6s":
        return ("bass.wav", "drums.wav", "guitar.wav", "other.wav", "piano.wav", "vocals.wav")
    return ("bass.wav", "drums.wav", "other.wav", "vocals.wav")


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def is_completed(out_root: Path, model: str, audio_path: Path, stems: tuple[str, ...]) -> bool:
    stem_dir = out_root / model / audio_path.stem
    return all((stem_dir / stem).exists() for stem in stems)


def sanitized_input(path: Path, staging_dir: Path) -> Path:
    clean_stem = " ".join(path.stem.split())
    if clean_stem == path.stem:
        return path
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / f"{clean_stem}{path.suffix.lower()}"
    if not staged.exists() or staged.stat().st_size != path.stat().st_size:
        shutil.copy2(path, staged)
    return staged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/demucs_stems"))
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mp3", action="store_true", help="Write mp3 stems instead of wav stems.")
    parser.add_argument("--two-stems", choices=["vocals", "drums", "bass", "other"], default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sanitize-names", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    files = discover_audio(args.input_dir)
    stems = expected_stems(args.model, args.two_stems)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    staging_dir = args.out_root / "_sanitized_inputs"
    if args.skip_existing:
        before = len(files)
        files = [
            path
            for path in files
            if not is_completed(args.out_root, args.model, path, stems)
            and not is_completed(args.out_root, args.model, sanitized_input(path, staging_dir) if args.sanitize_names else path, stems)
        ]
        print(f"skipping completed files: {before - len(files)}")
    if not files:
        print("No remaining files to separate.")
        return

    args.out_root.mkdir(parents=True, exist_ok=True)
    base_cmd = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        args.model,
        "-o",
        str(args.out_root),
        "-j",
        str(args.jobs),
    ]
    if args.mp3:
        base_cmd.append("--mp3")
    if args.two_stems:
        base_cmd.extend(["--two-stems", args.two_stems])

    failures = []
    for index, path in enumerate(files, start=1):
        demucs_input = sanitized_input(path, staging_dir) if args.sanitize_names else path
        cmd = [*base_cmd, str(demucs_input)]
        print(f"[{index}/{len(files)}] {path}")
        print("> " + " ".join(cmd))
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            failures.append((path, exc.returncode))
            print(f"FAILED: {path} (exit {exc.returncode})")
            if not args.continue_on_error:
                raise

    print(f"requested files: {len(files)}")
    print(f"failed files: {len(failures)}")
    for path, code in failures:
        print(f"  - {path} (exit {code})")
    print(f"out: {args.out_root}")


if __name__ == "__main__":
    main()
