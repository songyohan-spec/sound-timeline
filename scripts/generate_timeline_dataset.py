from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.synthesis import (
    SAMPLE_RATE,
    SourceSpec,
    apply_effects,
    apply_highpass_rise,
    apply_lowpass_sweep,
    apply_sidechain_pumping,
    db_to_amp,
    make_source,
    normalize,
    to_stereo,
    widen_mono,
)


@dataclass(frozen=True)
class LayerPlan:
    source: str
    role: str
    articulation: str
    melody: str
    start: float
    end: float
    gain_db: float
    effects: list[str]
    spatial_texture: list[str]


def prominence_from_gain(gain_db: float) -> str:
    if gain_db >= -8:
        return "high"
    if gain_db >= -15:
        return "medium"
    return "low"


def make_layer_audio(plan: LayerPlan, bpm: float) -> np.ndarray:
    duration = plan.end - plan.start
    spec = SourceSpec(plan.source, plan.role, plan.melody, plan.articulation)
    audio = make_source(spec, duration)

    static_effects = [e for e in plan.effects if e not in {"lowpass_sweep_opening", "highpass_rise", "sidechain_pumping"}]
    audio = apply_effects(audio, static_effects)

    if "lowpass_sweep_opening" in plan.effects:
        audio = apply_lowpass_sweep(audio)
    if "highpass_rise" in plan.effects:
        audio = apply_highpass_rise(audio)
    if "sidechain_pumping" in plan.effects:
        audio = apply_sidechain_pumping(audio, bpm=bpm)

    if "wide" in plan.spatial_texture:
        audio = widen_mono(audio)
    else:
        audio = to_stereo(audio)

    return audio * db_to_amp(plan.gain_db)


def choose_arrangement(total_duration: float) -> list[LayerPlan]:
    plans: list[LayerPlan] = []

    # A small, musically plausible 16-second alt-pop style sketch:
    # texture intro, bass/drums entry, short transition FX, hook-like lead.
    plans.append(
        LayerPlan(
            source="guitar_like",
            role="background_texture",
            articulation=random.choice(["plucked", "sustained", "pulsing"]),
            melody=random.choice(["motif_repeated", "melody_active"]),
            start=0.0,
            end=total_duration,
            gain_db=random.uniform(-16, -10),
            effects=random.sample(["reverb", "chorus", "lowpass_filter", "lowpass_sweep_opening"], k=2),
            spatial_texture=["wide", "washed_out"],
        )
    )

    plans.append(
        LayerPlan(
            source="bass",
            role="bass_foundation",
            articulation="sustained",
            melody=random.choice(["motif_repeated", "non_melodic"]),
            start=4.0,
            end=total_duration,
            gain_db=random.uniform(-10, -6),
            effects=random.sample(["distortion", "lowpass_filter", "sidechain_pumping"], k=2),
            spatial_texture=random.choice([["dry_close"], ["dry_close", "pumped"]]),
        )
    )

    plans.append(
        LayerPlan(
            source="synth",
            role=random.choice(["main_melody", "background_texture"]),
            articulation=random.choice(["chopped", "pulsing", "sustained"]),
            melody=random.choice(["melody_active", "motif_repeated"]),
            start=8.0,
            end=total_duration,
            gain_db=random.uniform(-13, -7),
            effects=random.sample(["reverb", "delay", "chorus", "sidechain_pumping"], k=2),
            spatial_texture=random.choice([["wide"], ["wide", "pumped"]]),
        )
    )

    plans.append(
        LayerPlan(
            source="noise_fx",
            role="transition_fx",
            articulation="swelling",
            melody="non_melodic",
            start=6.0,
            end=8.0,
            gain_db=random.uniform(-10, -6),
            effects=["highpass_rise", "reverb"],
            spatial_texture=["wide", "grainy"],
        )
    )

    if random.random() < 0.65:
        plans.append(
            LayerPlan(
                source="vocal_like",
                role="main_melody",
                articulation=random.choice(["sustained", "chopped"]),
                melody="melody_active",
                start=8.0,
                end=total_duration,
                gain_db=random.uniform(-12, -7),
                effects=random.sample(["reverb", "delay", "chorus", "distortion"], k=2),
                spatial_texture=random.choice([["dry_close"], ["wide"]]),
            )
        )

    return plans


def render_mix(index: int, out_dir: Path, duration: float, bpm: float) -> dict:
    mix = np.zeros((2, int(SAMPLE_RATE * duration)), dtype=np.float32)
    regions = []

    for plan in choose_arrangement(duration):
        layer = make_layer_audio(plan, bpm)
        start_sample = int(plan.start * SAMPLE_RATE)
        end_sample = min(start_sample + layer.shape[1], mix.shape[1])
        layer = layer[:, : end_sample - start_sample]
        mix[:, start_sample:end_sample] += layer
        regions.append(
            {
                "start": plan.start,
                "end": plan.end,
                "source": plan.source,
                "role": plan.role,
                "articulation": plan.articulation,
                "effects": plan.effects,
                "spatial_texture": plan.spatial_texture,
                "melody": plan.melody,
                "gain_db": round(plan.gain_db, 2),
                "prominence": prominence_from_gain(plan.gain_db),
            }
        )

    mix = normalize(mix, peak=0.92)
    filename = f"mix_{index:06d}.wav"
    sf.write(out_dir / "mixes" / filename, mix.T, SAMPLE_RATE)

    return {
        "file": f"mixes/{filename}",
        "duration": duration,
        "bpm": bpm,
        "key": "C minor",
        "regions": regions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=Path("data/timeline_synthetic"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mixes").mkdir(parents=True, exist_ok=True)

    metadata_path = args.out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for index in range(args.count):
            row = render_mix(index, args.out, args.duration, args.bpm)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {args.count} timeline mixes")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

