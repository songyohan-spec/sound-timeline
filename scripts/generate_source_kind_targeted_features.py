from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Bitcrush, Chorus, Distortion, HighpassFilter, LowpassFilter, Pedalboard, Reverb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


META_FIELDS = [
    "file",
    "labels",
    "groups",
    "primary_label",
    "primary_group",
    "source_file",
    "source_index",
    "duration",
    "training_labels",
]


TARGET_COUNTS = {
    "synth_pad_or_wash": 260,
    "supersaw_or_bright_synth_stack": 260,
    "digital_synth_lead": 260,
    "bitcrushed_or_aliasing_synth": 260,
    "arpeggio_or_sequence_synth": 260,
    "fuzzy_distorted_synth": 180,
    "wavetable_noise_synth": 180,
    "granular_or_resampled_synth": 180,
    "synth_pluck_or_bell": 180,
    "vocal_synth_hybrid": 180,
    "formant_or_vocoder_vocal": 180,
    "sampled_loop_texture": 120,
    "filtered_or_muffled_loop": 120,
    "guitar_or_plucked_loop": 120,
    "washed_guitar_or_strings": 120,
    "synth_bass": 120,
    "sidechained_bass_pulse": 120,
}


def group_for(label: str) -> str:
    if "vocal" in label or "vocoder" in label or "formant" in label:
        return "vocal_texture"
    if "bass" in label or "808" in label or "sidechain" in label:
        return "bass"
    if "guitar" in label or "string" in label:
        return "guitar_strings"
    if "sample" in label or "loop" in label:
        return "sample_loop"
    return "synth"


def normalize(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.astype(np.float32)
    return np.stack([audio, audio], axis=0).astype(np.float32)


def widen(audio: np.ndarray, amount: float = 0.5, delay: int = 17) -> np.ndarray:
    right = np.roll(audio, delay) * amount + audio * (1.0 - amount)
    return normalize(np.stack([audio, right], axis=0))


def sine(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    if isinstance(freq, np.ndarray):
        phase = np.cumsum(freq) / max(1, len(freq))
        return np.sin(2.0 * math.pi * phase)
    return np.sin(2.0 * math.pi * freq * t)


def saw(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    if isinstance(freq, np.ndarray):
        phase = np.cumsum(freq) / 22_050.0
        return 2.0 * (phase % 1.0) - 1.0
    return 2.0 * ((freq * t) % 1.0) - 1.0


def square(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.sign(sine(freq, t))


def env_adsr(n: int, sr: int, attack: float, release: float, sustain: float = 0.8) -> np.ndarray:
    env = np.ones(n, dtype=np.float32) * sustain
    a = min(n, int(sr * attack))
    r = min(n, int(sr * release))
    if a:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r:
        env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def board(audio: np.ndarray, sr: int, pedals: list) -> np.ndarray:
    return normalize(Pedalboard(pedals)(audio.astype(np.float32), sr))


def rhythm_gate(t: np.ndarray, rate: float, duty: float) -> np.ndarray:
    return (((t * rate) % 1.0) < duty).astype(np.float32)


def sidechain(audio: np.ndarray, sr: int, depth: float = 0.78, bpm: float = 120.0) -> np.ndarray:
    t = np.arange(len(audio)) / sr
    beat = 60.0 / bpm
    phase = (t % beat) / beat
    recover = np.clip(phase / 0.72, 0.0, 1.0)
    curve = 1.0 - depth * (1.0 - recover) ** 2.1
    return normalize(audio * curve)


def vocalish(t: np.ndarray, root: float, rng: np.random.Generator, breath: float = 0.04) -> np.ndarray:
    out = np.zeros_like(t, dtype=np.float32)
    for harmonic, gain in [(1, 1.0), (2, 0.55), (3, 0.27), (4, 0.12), (5, 0.06)]:
        out += gain * sine(root * harmonic, t)
    out += rng.normal(0.0, breath, len(t)).astype(np.float32)
    return normalize(out)


def render(label: str, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    root = float(rng.choice([98.0, 110.0, 130.81, 146.83, 164.81, 196.0, 220.0]))
    n = len(t)

    if label == "synth_pad_or_wash":
        detunes = [0.992, 0.997, 1.0, 1.004, 1.009, 2.0]
        audio = sum(saw(root * d, t + rng.uniform(0, 0.01)) for d in detunes) / len(detunes)
        audio *= env_adsr(n, sr, rng.uniform(0.32, 0.75), rng.uniform(0.45, 0.9), 0.72)
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(1100, 2800)), Chorus(rate_hz=0.5, depth=0.75, mix=0.55), Reverb(room_size=0.88, wet_level=0.42, dry_level=0.65)])
        return widen(audio, 0.76)

    if label == "supersaw_or_bright_synth_stack":
        detunes = [0.985, 0.993, 1.0, 1.007, 1.015, 1.99, 2.01]
        audio = sum(saw(root * rng.choice([1.0, 2.0]) * d, t) for d in detunes) / len(detunes)
        audio *= env_adsr(n, sr, rng.uniform(0.04, 0.22), rng.uniform(0.18, 0.45), 0.82)
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=120), LowpassFilter(cutoff_frequency_hz=rng.uniform(3500, 7200)), Chorus(rate_hz=0.9, depth=0.7, mix=0.45), Reverb(room_size=0.55, wet_level=0.18)])
        return widen(audio, 0.66)

    if label == "digital_synth_lead":
        vib = rng.uniform(2.0, 8.0) * np.sin(2.0 * math.pi * rng.uniform(4.5, 7.0) * t)
        audio = 0.6 * square(root * rng.choice([2.0, 3.0]) + vib, t) + 0.4 * saw(root * rng.choice([2.0, 2.5]) + vib, t)
        audio *= env_adsr(n, sr, 0.006, 0.08, 0.78)
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=220), LowpassFilter(cutoff_frequency_hz=rng.uniform(4300, 8500)), Chorus(rate_hz=1.4, depth=0.25, mix=0.18)])
        return stereo(audio)

    if label == "bitcrushed_or_aliasing_synth":
        notes = np.array([root * 2.0, root * 2.5, root * 3.0, root * 4.0])
        freq = notes[np.floor((t * rng.choice([4.0, 6.0, 8.0])) % len(notes)).astype(int)]
        audio = 0.7 * saw(freq, t) + 0.3 * square(freq * 0.5, t)
        audio *= rhythm_gate(t, rng.choice([4.0, 8.0, 12.0]), rng.uniform(0.35, 0.7))
        audio = board(audio, sr, [Distortion(drive_db=rng.uniform(8, 18)), Bitcrush(bit_depth=int(rng.choice([4, 5, 6, 7]))), HighpassFilter(cutoff_frequency_hz=260)])
        return stereo(audio)

    if label == "arpeggio_or_sequence_synth":
        notes = np.array([root, root * 1.25, root * 1.5, root * 2.0, root * 2.5])
        freq = notes[np.floor((t * rng.choice([6.0, 8.0, 10.0, 12.0])) % len(notes)).astype(int)]
        audio = 0.55 * square(freq, t) + 0.45 * sine(freq * 2.0, t)
        audio *= rhythm_gate(t, rng.choice([6.0, 8.0, 12.0]), rng.uniform(0.35, 0.58))
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(2200, 6200)), Chorus(rate_hz=1.2, depth=0.35, mix=0.22), Reverb(room_size=0.32, wet_level=0.12)])
        return stereo(audio)

    if label == "fuzzy_distorted_synth":
        audio = saw(root, t) + 0.16 * rng.normal(0.0, 1.0, n)
        audio *= env_adsr(n, sr, 0.02, 0.22, 0.75)
        audio = board(audio, sr, [Distortion(drive_db=rng.uniform(7, 14)), LowpassFilter(cutoff_frequency_hz=rng.uniform(1000, 2800))])
        return stereo(audio)

    if label in {"wavetable_noise_synth", "granular_or_resampled_synth"}:
        grains = rng.normal(0.0, 0.25, n)
        carrier = 0.38 * saw(root * rng.choice([0.5, 1.0, 2.0]), t) + 0.22 * square(root * 1.5, t)
        gate = rhythm_gate(t, rng.choice([8.0, 12.0, 16.0]), rng.uniform(0.18, 0.42)) if label == "granular_or_resampled_synth" else 1.0
        audio = (carrier + grains) * gate
        audio = board(audio, sr, [Distortion(drive_db=8), Bitcrush(bit_depth=int(rng.choice([6, 7, 8]))), Reverb(room_size=0.48, wet_level=0.2)])
        return widen(audio, 0.45)

    if label == "synth_pluck_or_bell":
        partials = sine(root * 3.0, t) + 0.6 * sine(root * 5.04, t) + 0.24 * sine(root * 7.5, t)
        audio = partials * np.exp(-t * rng.uniform(3.2, 7.5))
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=260), Reverb(room_size=0.38, wet_level=0.16)])
        return stereo(audio)

    if label in {"vocal_synth_hybrid", "formant_or_vocoder_vocal"}:
        notes = np.array([root * 1.5, root * 2.0, root * 2.25, root * 3.0])
        freq = notes[np.floor((t * rng.choice([3.0, 4.0, 6.0])) % len(notes)).astype(int)]
        voice = vocalish(t, freq, rng, breath=0.03)
        edge = 0.32 * saw(root * 2.0, t) + 0.2 * square(root, t)
        audio = 0.7 * voice + edge
        if label == "formant_or_vocoder_vocal":
            audio *= rhythm_gate(t, rng.choice([4.0, 6.0, 8.0]), rng.uniform(0.45, 0.68))
            pedals = [HighpassFilter(cutoff_frequency_hz=260), LowpassFilter(cutoff_frequency_hz=3200), Chorus(rate_hz=1.5, depth=0.45, mix=0.38)]
        else:
            pedals = [LowpassFilter(cutoff_frequency_hz=3600), Chorus(rate_hz=1.0, depth=0.4, mix=0.32), Distortion(drive_db=3)]
        return widen(board(audio, sr, pedals), 0.42)

    if label == "synth_bass":
        audio = sine(rng.choice([43.65, 49.0, 55.0, 65.41]), t) + 0.18 * saw(root * 0.5, t)
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=650)])
        return stereo(audio)

    if label == "sidechained_bass_pulse":
        audio = sine(55.0, t) + 0.24 * saw(110.0, t)
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=800)])
        return stereo(sidechain(audio, sr, 0.86))

    if label == "guitar_or_plucked_loop":
        pulse = np.exp(-6.5 * (t % 0.5))
        audio = (sine(root * 1.5, t) + 0.3 * sine(root * 3.0, t)) * pulse
        return stereo(board(audio, sr, [LowpassFilter(cutoff_frequency_hz=1800)]))

    if label == "washed_guitar_or_strings":
        audio = sine(root * 1.5, t) * env_adsr(n, sr, 0.12, 0.55, 0.68)
        return widen(board(audio, sr, [Chorus(rate_hz=1.1, depth=0.82, mix=0.6), Reverb(room_size=0.86, wet_level=0.48)]), 0.7)

    if label in {"sampled_loop_texture", "filtered_or_muffled_loop"}:
        audio = 0.38 * saw(root, t) + 0.32 * sine(root * 1.5, t) + 0.12 * rng.normal(0.0, 1.0, n)
        audio *= rhythm_gate(t, rng.choice([3.0, 4.0, 6.0]), rng.uniform(0.45, 0.72))
        cutoff = rng.uniform(700, 1600) if label == "filtered_or_muffled_loop" else rng.uniform(1400, 3400)
        return widen(board(audio, sr, [LowpassFilter(cutoff_frequency_hz=cutoff), Reverb(room_size=0.32, wet_level=0.12)]), 0.35)

    audio = rng.normal(0.0, 0.12, n)
    return stereo(normalize(audio))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-data", type=Path, default=Path("data/source_kind_targeted_v1"))
    parser.add_argument("--out-features", type=Path, default=Path("outputs/source_kind_targeted_features_v1.csv"))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--seed", type=int, default=526)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    audio_dir = args.out_data / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    index = 0
    for label, count in TARGET_COUNTS.items():
        group = group_for(label)
        for local_idx in range(count):
            audio = render(label, args.duration, args.sample_rate, rng)
            out_file = audio_dir / f"sourcekind_{index:06d}_{label}.wav"
            sf.write(out_file, audio.T if audio.ndim == 2 else audio, args.sample_rate)
            stats = audio_stats(out_file, quality=args.quality)
            row = {
                "file": out_file.as_posix(),
                "labels": label,
                "groups": group,
                "primary_label": label,
                "primary_group": group,
                "source_file": f"targeted:{label}",
                "source_index": local_idx,
                "duration": args.duration,
                "training_labels": label,
            }
            row.update(stats)
            rows.append(row)
            index += 1
        print(f"{label}: {count}")

    stat_fields = sorted(key for key in rows[0] if key not in META_FIELDS)
    args.out_features.parent.mkdir(parents=True, exist_ok=True)
    with args.out_features.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=META_FIELDS + stat_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.out_features}")


if __name__ == "__main__":
    main()
