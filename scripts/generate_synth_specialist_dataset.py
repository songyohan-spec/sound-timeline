from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Bitcrush, Chorus, Distortion, HighpassFilter, LowpassFilter, Pedalboard, Reverb

from generate_reference_synthetic_dataset import make_audio, normalize, sidechain, rhythm_gate


SYNTH_RECIPES = {
    "synth_pad_wash": ["lush_synth_pad"],
    "supersaw_stack": ["lush_synth_pad"],
    "digital_synth_lead": ["syrupy_video_game_synth_melody", "bitcrushed_synth_lead"],
    "bitcrushed_synth_lead": ["bitcrushed_synth_lead"],
    "synth_pluck_bell": ["syrupy_video_game_synth_melody", "synth_flute_or_recorder_like_lead"],
    "arpeggio_sequence": ["syrupy_video_game_synth_melody"],
    "granular_texture": ["noisy_wavetable_texture", "fuzzy_diy_synth_texture"],
    "wavetable_noise": ["noisy_wavetable_texture"],
    "fuzzy_lofi_synth": ["fuzzy_diy_synth_texture"],
    "synth_flute_pipe": ["synth_flute_or_recorder_like_lead"],
    "vocal_synth_hybrid": ["vocal_synth_hybrid"],
    "formant_vocoder": ["vocal_synth_hybrid", "hard_tuned_vocal"],
    "synth_bass": ["sub_bass", "pulsing_sidechain_bass"],
    "sidechained_synth_bass": ["pulsing_sidechain_bass"],
    "sub_808_synth_bass": ["sub_bass", "distorted_808_bass"],
}

NEGATIVE_RECIPES = {
    "not_synth_vocal": ["processed_lead_vocal", "pitched_vocal_chop", "breathy_vocal_pad", "stacked_harmony_vocal"],
    "not_synth_guitar": ["filtered_guitar_loop", "washed_chorus_guitar", "distorted_guitar_texture"],
    "not_synth_drums": ["trap_drum_pattern", "glitch_percussion"],
    "not_synth_sample_loop": ["filtered_sample_loop", "unknown_hybrid_loop"],
}


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def board(audio: np.ndarray, sr: int, pedals: list) -> np.ndarray:
    return normalize(Pedalboard(pedals)(audio.astype(np.float32), sr))


def transform(label: str, audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    mono = to_mono(audio)
    if label == "supersaw_stack":
        detuned = mono + 0.55 * np.roll(mono, 5) + 0.42 * np.roll(mono, 11)
        return board(detuned, sr, [Chorus(rate_hz=0.55, depth=0.85, mix=0.62), Reverb(room_size=0.74, wet_level=0.25)])
    if label == "synth_pluck_bell":
        plucked = mono * rhythm_gate(np.arange(len(mono)) / sr, rng.choice([5.0, 6.0, 8.0]), 0.24)
        return board(plucked, sr, [HighpassFilter(cutoff_frequency_hz=320), Reverb(room_size=0.32, wet_level=0.12)])
    if label == "arpeggio_sequence":
        seq = mono * rhythm_gate(np.arange(len(mono)) / sr, rng.choice([6.0, 8.0, 12.0]), 0.38)
        return board(seq, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(1800, 5200)), Chorus(rate_hz=1.0, depth=0.28, mix=0.18)])
    if label == "granular_texture":
        grains = mono.copy()
        for shift in rng.integers(200, 2400, size=5):
            grains += rng.uniform(0.08, 0.18) * np.roll(mono, int(shift))
        grains += rng.normal(0.0, 0.035, len(mono))
        return board(grains, sr, [Bitcrush(bit_depth=int(rng.choice([7, 8, 9]))), Reverb(room_size=0.48, wet_level=0.18)])
    if label == "wavetable_noise":
        noisy = mono + rng.normal(0.0, 0.10, len(mono))
        return board(noisy, sr, [Distortion(drive_db=rng.uniform(4, 10)), LowpassFilter(cutoff_frequency_hz=rng.uniform(1800, 4200))])
    if label == "formant_vocoder":
        return board(mono, sr, [HighpassFilter(cutoff_frequency_hz=420), LowpassFilter(cutoff_frequency_hz=3400), Chorus(rate_hz=1.7, depth=0.45, mix=0.45), Bitcrush(bit_depth=9)])
    if label == "sidechained_synth_bass":
        return sidechain(board(mono, sr, [LowpassFilter(cutoff_frequency_hz=720)]), sr, depth=0.92)
    if label == "sub_808_synth_bass":
        return board(mono, sr, [LowpassFilter(cutoff_frequency_hz=520), Distortion(drive_db=rng.uniform(2, 8))])
    if label == "bitcrushed_synth_lead":
        return board(mono, sr, [Bitcrush(bit_depth=int(rng.choice([5, 6, 7]))), Distortion(drive_db=rng.uniform(6, 14))])
    if label == "fuzzy_lofi_synth":
        return board(mono + rng.normal(0.0, 0.04, len(mono)), sr, [LowpassFilter(cutoff_frequency_hz=2100), Distortion(drive_db=6)])
    return mono


def stereoize(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.45:
        return np.stack([audio, audio], axis=0)
    delay = int(rng.integers(4, 18))
    right = 0.72 * np.roll(audio, delay) + 0.28 * audio
    return normalize(np.stack([audio, right], axis=0))


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synth_specialist_v1"))
    parser.add_argument("--count-per-label", type=int, default=70)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=527)
    parser.add_argument("--include-negatives", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = dict(SYNTH_RECIPES)
    if args.include_negatives:
        labels.update(NEGATIVE_RECIPES)

    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    idx = 0
    for label, bases in labels.items():
        family = "not_synth" if label.startswith("not_synth") else ("synth_bass" if "bass" in label else "synth")
        for source_index in range(args.count_per_label):
            base = str(rng.choice(bases))
            audio = make_audio(base, args.duration, args.sample_rate, rng)
            audio = transform(label, audio, args.sample_rate, rng)
            audio = stereoize(normalize(audio), rng)
            out_file = audio_dir / f"synthspec_{idx:06d}_{label}.wav"
            sf.write(out_file, audio.T, args.sample_rate)
            rows.append(
                {
                    "file": out_file.relative_to(args.out).as_posix(),
                    "label": label,
                    "family": family,
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
