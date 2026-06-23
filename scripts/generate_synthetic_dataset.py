from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.synthesis import SAMPLE_RATE, SourceSpec, apply_effects, make_source, widen_mono


SOURCES = ["synth", "bass", "guitar_like", "vocal_like", "noise_fx"]
ROLES_BY_SOURCE = {
    "synth": ["main_melody", "background_texture"],
    "bass": ["bass_foundation"],
    "guitar_like": ["main_melody", "background_texture", "rhythmic_layer"],
    "vocal_like": ["main_melody", "background_texture"],
    "noise_fx": ["transition_fx", "background_texture"],
}
ARTICULATIONS = ["sustained", "plucked", "chopped", "swelling", "pulsing"]
EFFECTS = ["reverb", "delay", "distortion", "lowpass_filter", "highpass_filter", "bitcrush", "chorus"]
MELODY = ["melody_active", "motif_repeated", "non_melodic"]


def choose_effects(source: str, role: str) -> list[str]:
    max_effects = 3
    weights = EFFECTS.copy()
    if source == "noise_fx" or role == "transition_fx":
        weights += ["reverb", "highpass_filter"]
    if source == "bass":
        weights += ["distortion", "lowpass_filter"]
    if source in {"synth", "guitar_like"}:
        weights += ["chorus", "reverb", "lowpass_filter"]
    count = random.randint(0, max_effects)
    return sorted(set(random.sample(weights, k=count)))


def generate_one(index: int, out_dir: Path, duration: float) -> dict:
    source = random.choice(SOURCES)
    role = random.choice(ROLES_BY_SOURCE[source])
    articulation = random.choice(ARTICULATIONS)
    melody = "non_melodic" if role in {"transition_fx", "background_texture"} and random.random() < 0.55 else random.choice(MELODY)
    effects = choose_effects(source, role)

    spec = SourceSpec(source=source, role=role, melody=melody, articulation=articulation)
    audio = make_source(spec, duration)
    audio = apply_effects(audio, effects)

    spatial_texture = []
    if random.random() < 0.35 or "chorus" in effects or "reverb" in effects:
        audio = widen_mono(audio)
        spatial_texture.append("wide")
    else:
        spatial_texture.append("dry_close")

    if "reverb" in effects and "lowpass_filter" in effects:
        spatial_texture.append("washed_out")
    if "bitcrush" in effects or source == "noise_fx":
        spatial_texture.append("grainy")

    filename = f"sample_{index:06d}.wav"
    sf.write(out_dir / "audio" / filename, audio.T if audio.ndim == 2 else audio, SAMPLE_RATE)

    return {
        "file": f"audio/{filename}",
        "duration": duration,
        "source": source,
        "role": role,
        "articulation": articulation,
        "effects": effects,
        "spatial_texture": spatial_texture,
        "melody": melody,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)

    metadata_path = args.out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for index in range(args.count):
            row = generate_one(index, args.out, args.duration)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {args.count} samples")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()

