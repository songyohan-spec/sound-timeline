from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from generate_reference_synthetic_dataset import make_audio, normalize
from generate_synth_specialist_dataset import NEGATIVE_RECIPES, SYNTH_RECIPES, stereoize, transform


DEFAULT_LABEL_COUNTS = {
    "synth_pad_wash": 140,
    "supersaw_stack": 140,
    "digital_synth_lead": 120,
    "bitcrushed_synth_lead": 100,
    "arpeggio_sequence": 100,
    "wavetable_noise": 100,
    "fuzzy_lofi_synth": 100,
    "vocal_synth_hybrid": 140,
    "formant_vocoder": 110,
    "synth_flute_pipe": 80,
    "not_synth_vocal": 80,
    "not_synth_guitar": 80,
    "not_synth_drums": 80,
    "not_synth_sample_loop": 80,
}


def family_for(label: str) -> str:
    if label.startswith("not_synth"):
        return "not_synth"
    if "bass" in label:
        return "synth_bass"
    return "synth"


def read_counts(value: str | None, path: Path | None) -> dict[str, int]:
    if path:
        return {str(k): int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    if value:
        out = {}
        for item in value.split(","):
            if not item.strip():
                continue
            label, count = item.split(":", 1)
            out[label.strip()] = int(count)
        return out
    return dict(DEFAULT_LABEL_COUNTS)


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synth_specialist_targeted_v1"))
    parser.add_argument("--label-counts", default="")
    parser.add_argument("--label-counts-json", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=931)
    args = parser.parse_args()

    counts = read_counts(args.label_counts, args.label_counts_json)
    recipes = {**SYNTH_RECIPES, **NEGATIVE_RECIPES}
    unknown = sorted(set(counts) - set(recipes))
    if unknown:
        raise SystemExit(f"Unknown labels in counts: {', '.join(unknown)}")

    rng = np.random.default_rng(args.seed)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for label, count in counts.items():
        bases = recipes[label]
        for source_index in range(count):
            base = str(rng.choice(bases))
            audio = make_audio(base, args.duration, args.sample_rate, rng)
            audio = transform(label, audio, args.sample_rate, rng)
            audio = stereoize(normalize(audio), rng)
            out_file = audio_dir / f"targeted_{idx:06d}_{label}.wav"
            sf.write(out_file, audio.T, args.sample_rate)
            rows.append(
                {
                    "file": out_file.relative_to(args.out).as_posix(),
                    "label": label,
                    "family": family_for(label),
                    "base_label": base,
                    "duration": args.duration,
                    "source_index": source_index,
                }
            )
            idx += 1

    write_jsonl(rows, args.out / "metadata.jsonl")
    print(f"clips: {len(rows)}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
