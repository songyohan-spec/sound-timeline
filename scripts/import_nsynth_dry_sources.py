from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path


NSYNTH_TEST_URL = "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz"
DEFAULT_FAMILY_MAP = {
    "bass": "bass",
    "guitar": "guitar_like",
    "synth_lead": "synth",
    "vocal": "vocal_like",
}


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"using existing archive: {out_path}")
        return

    print(f"downloading: {url}")
    print(f"to: {out_path}")
    with urllib.request.urlopen(url) as response, out_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def find_member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    for member in tar.getmembers():
        if member.name.endswith(suffix):
            return member
    raise FileNotFoundError(f"Could not find {suffix} inside archive")


def load_examples(tar: tarfile.TarFile) -> dict:
    member = find_member(tar, "examples.json")
    extracted = tar.extractfile(member)
    if extracted is None:
        raise RuntimeError("Could not read examples.json from archive")
    return json.loads(extracted.read().decode("utf-8"))


def map_nsynth_row(row: dict, include_electronic_as_synth: bool) -> str | None:
    family = row.get("instrument_family_str")
    target = DEFAULT_FAMILY_MAP.get(family)
    if target is not None:
        return target

    source = row.get("instrument_source_str")
    if include_electronic_as_synth and source in {"electronic", "synthetic"}:
        return "synth"
    return None


def choose_examples(examples: dict, max_per_family: int, seed: int, include_electronic_as_synth: bool) -> list[tuple[str, dict, str]]:
    rng = random.Random(seed)
    grouped: dict[str, list[tuple[str, dict]]] = {target: [] for target in DEFAULT_FAMILY_MAP.values()}

    for note_id, row in examples.items():
        target = map_nsynth_row(row, include_electronic_as_synth)
        if target is None:
            continue
        grouped[target].append((note_id, row))

    selected: list[tuple[str, dict, str]] = []
    for target, items in grouped.items():
        rng.shuffle(items)
        take = items[:max_per_family]
        print(f"{target:12s}: selected {len(take)} / available {len(items)}")
        for note_id, row in take:
            selected.append((note_id, row, target))
    return selected


def extract_selected(tar: tarfile.TarFile, selected: list[tuple[str, dict, str]], out_root: Path, manifest_path: Path) -> None:
    members_by_name = {member.name: member for member in tar.getmembers() if member.isfile()}
    out_root.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for note_id, row, target in selected:
            suffix = f"audio/{note_id}.wav"
            member = next((m for name, m in members_by_name.items() if name.endswith(suffix)), None)
            if member is None:
                print(f"warning: missing audio for {note_id}")
                continue

            target_dir = out_root / target
            target_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"nsynth_{note_id}.wav"
            out_path = target_dir / out_name

            src = tar.extractfile(member)
            if src is None:
                print(f"warning: could not extract {member.name}")
                continue
            with out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

            manifest.write(
                json.dumps(
                    {
                        "file": str(out_path.as_posix()),
                        "source_family": target,
                        "dataset": "NSynth test",
                        "note_id": note_id,
                        "instrument_family_str": row.get("instrument_family_str"),
                        "instrument_source_str": row.get("instrument_source_str"),
                        "instrument_str": row.get("instrument_str"),
                        "pitch": row.get("pitch"),
                        "velocity": row.get("velocity"),
                        "qualities_str": row.get("qualities_str", []),
                        "license": "CC BY 4.0",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/raw/nsynth-test.jsonwav.tar.gz"))
    parser.add_argument("--out-root", type=Path, default=Path("data/dry_sources"))
    parser.add_argument("--manifest", type=Path, default=Path("data/dry_sources/nsynth_manifest.jsonl"))
    parser.add_argument("--max-per-family", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--include-electronic-as-synth", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.download:
        download_file(NSYNTH_TEST_URL, args.archive)

    if not args.archive.exists():
        raise SystemExit(
            f"Archive not found: {args.archive}\n"
            "Run with --download or manually place nsynth-test.jsonwav.tar.gz at that path."
        )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, "r:gz") as tar:
        examples = load_examples(tar)
        selected = choose_examples(examples, args.max_per_family, args.seed, args.include_electronic_as_synth)
        extract_selected(tar, selected, args.out_root, args.manifest)

    print(f"wrote dry sources to: {args.out_root}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
