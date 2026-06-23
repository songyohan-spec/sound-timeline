from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_reference_synthetic_dataset import DEFAULT_LABELS, GROUP_BY_LABEL, make_audio, normalize


VOCAL_FOCUS_LABELS = {
    "processed_lead_vocal",
    "hard_tuned_vocal",
    "pitched_vocal_chop",
    "breathy_vocal_pad",
    "stacked_harmony_vocal",
    "vocal_synth_hybrid",
}


def weighted_choice(labels: list[str], rng: random.Random, vocal_focus: float) -> str:
    weights = []
    for label in labels:
        group = GROUP_BY_LABEL.get(label, "unknown")
        weight = 1.0
        if label in VOCAL_FOCUS_LABELS:
            weight *= vocal_focus
        if group == "vocals":
            weight *= 1.35
        weights.append(weight)
    return rng.choices(labels, weights=weights, k=1)[0]


def choose_labels(labels: list[str], rng: random.Random, max_layers: int, vocal_focus: float) -> list[str]:
    layer_count = rng.choices([1, 2, 3], weights=[0.35, 0.45, 0.20], k=1)[0]
    layer_count = min(layer_count, max_layers, len(labels))

    # Encourage realistic combinations: one foreground plus optional support.
    foreground = weighted_choice(labels, rng, vocal_focus)
    selected = [foreground]
    foreground_group = GROUP_BY_LABEL.get(foreground, "unknown")
    support_pool = [label for label in labels if label != foreground]
    if foreground_group in {"vocals", "synth", "sampled_loop"}:
        support_pool = sorted(set(support_pool + [label for label in labels if GROUP_BY_LABEL.get(label) in {"bass", "drums"}]))
    while len(selected) < layer_count and support_pool:
        candidate = weighted_choice(support_pool, rng, max(1.0, vocal_focus * 0.6))
        if candidate not in selected:
            selected.append(candidate)
    return selected


def mix_layers(label_list: list[str], duration: float, sr: int, np_rng: np.random.Generator) -> np.ndarray:
    layers = []
    for idx, label in enumerate(label_list):
        audio = make_audio(label, duration, sr, np_rng)
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)
        gain = np_rng.uniform(0.45, 0.95) if idx == 0 else np_rng.uniform(0.22, 0.65)
        layers.append(audio * gain)
    mix = np.sum(np.stack(layers, axis=0), axis=0)
    return normalize(mix, peak=0.9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/reference_element_mixture_v1"))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--vocal-focus", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=91)
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    py_rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(args.count):
        selected = choose_labels(labels, py_rng, args.max_layers, args.vocal_focus)
        audio = mix_layers(selected, args.duration, args.sample_rate, np_rng)
        out_file = audio_dir / f"refmix_{idx:06d}.wav"
        sf.write(out_file, audio.T, args.sample_rate)
        groups = sorted({GROUP_BY_LABEL.get(label, "unknown") for label in selected})
        rows.append(
            {
                "file": out_file.relative_to(args.out).as_posix(),
                "duration": args.duration,
                "labels": selected,
                "groups": groups,
                "primary_label": selected[0],
                "primary_group": GROUP_BY_LABEL.get(selected[0], "unknown"),
                "source_file": "generated_mixture",
                "source_index": idx,
            }
        )

    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"clips: {len(rows)}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
