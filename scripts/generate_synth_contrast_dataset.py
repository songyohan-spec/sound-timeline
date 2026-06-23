from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Bitcrush, Chorus, Distortion, HighpassFilter, LowpassFilter, Pedalboard, Reverb


COUNTS = {
    "bitcrushed_synth_lead": 360,
    "digital_synth_lead": 220,
    "granular_texture": 260,
    "wavetable_noise": 240,
    "fuzzy_lofi_synth": 220,
    "synth_pluck_bell": 160,
    "arpeggio_sequence": 160,
    "not_synth_drums": 120,
    "not_synth_sample_loop": 120,
}


def normalize(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def stereoize(audio: np.ndarray, rng: np.random.Generator, wide_prob: float = 0.35) -> np.ndarray:
    if rng.random() > wide_prob:
        return np.stack([audio, audio], axis=0).astype(np.float32)
    delay = int(rng.integers(5, 22))
    right = 0.55 * np.roll(audio, delay) + 0.45 * audio
    return normalize(np.stack([audio, right], axis=0))


def sine(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    if isinstance(freq, np.ndarray):
        phase = np.cumsum(freq) / 22_050.0
        return np.sin(2.0 * math.pi * phase)
    return np.sin(2.0 * math.pi * freq * t)


def saw(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    if isinstance(freq, np.ndarray):
        phase = np.cumsum(freq) / 22_050.0
        return 2.0 * (phase % 1.0) - 1.0
    return 2.0 * ((freq * t) % 1.0) - 1.0


def square(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.sign(sine(freq, t))


def gate(t: np.ndarray, rate: float, duty: float) -> np.ndarray:
    return (((t * rate) % 1.0) < duty).astype(np.float32)


def env(n: int, sr: int, attack: float, release: float, sustain: float = 0.8) -> np.ndarray:
    out = np.ones(n, dtype=np.float32) * sustain
    a = min(n, int(attack * sr))
    r = min(n, int(release * sr))
    if a:
        out[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if r:
        out[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return out


def board(audio: np.ndarray, sr: int, pedals: list) -> np.ndarray:
    return normalize(Pedalboard(pedals)(audio.astype(np.float32), sr))


def grain_cloud(source: np.ndarray, rng: np.random.Generator, grain_count: int, min_len: int, max_len: int) -> np.ndarray:
    out = np.zeros_like(source)
    n = len(source)
    for _ in range(grain_count):
        length = int(rng.integers(min_len, max_len))
        if length >= n:
            continue
        src = int(rng.integers(0, n - length))
        dst = int(rng.integers(0, n - length))
        window = np.hanning(length).astype(np.float32)
        out[dst : dst + length] += source[src : src + length] * window * rng.uniform(0.28, 0.85)
    return normalize(out)


def render(label: str, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, duration, int(duration * sr), endpoint=False)
    n = len(t)
    root = float(rng.choice([98.0, 110.0, 130.81, 146.83, 164.81, 196.0, 220.0]))

    if label == "bitcrushed_synth_lead":
        notes = np.array([root * 2.0, root * 2.5, root * 3.0, root * 4.0])
        freq = notes[np.floor((t * rng.choice([4.0, 6.0, 8.0, 12.0])) % len(notes)).astype(int)]
        raw = 0.72 * saw(freq, t) + 0.28 * square(freq * rng.choice([0.5, 1.0]), t)
        raw *= gate(t, rng.choice([4.0, 8.0, 12.0]), rng.uniform(0.42, 0.72))
        audio = board(
            raw,
            sr,
            [
                HighpassFilter(cutoff_frequency_hz=rng.uniform(180, 420)),
                Distortion(drive_db=rng.uniform(8, 18)),
                Bitcrush(bit_depth=int(rng.choice([3, 4, 5, 6]))),
                LowpassFilter(cutoff_frequency_hz=rng.uniform(3200, 7800)),
            ],
        )
        return stereoize(audio, rng, 0.24)

    if label == "digital_synth_lead":
        vib = rng.uniform(1.5, 6.5) * np.sin(2.0 * math.pi * rng.uniform(4.0, 7.0) * t)
        audio = 0.52 * square(root * rng.choice([2.0, 2.5, 3.0]) + vib, t) + 0.48 * saw(root * 2.0 + vib, t)
        audio *= env(n, sr, rng.uniform(0.003, 0.025), rng.uniform(0.05, 0.16), 0.78)
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=180), LowpassFilter(cutoff_frequency_hz=rng.uniform(4500, 9000)), Chorus(rate_hz=1.3, depth=0.22, mix=0.16)])
        return stereoize(audio, rng, 0.22)

    if label == "granular_texture":
        carrier = 0.42 * saw(root * rng.choice([0.5, 1.0, 2.0]), t) + 0.25 * sine(root * 2.0, t)
        carrier += rng.normal(0.0, 0.08, n).astype(np.float32)
        grains = grain_cloud(carrier, rng, grain_count=int(rng.integers(28, 64)), min_len=int(0.012 * sr), max_len=int(0.075 * sr))
        audio = 0.55 * grains + 0.18 * carrier
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=rng.uniform(120, 420)), Reverb(room_size=rng.uniform(0.42, 0.72), wet_level=rng.uniform(0.16, 0.34)), Chorus(rate_hz=0.9, depth=0.45, mix=0.25)])
        return stereoize(audio, rng, 0.65)

    if label == "wavetable_noise":
        mod = 1.0 + 0.04 * np.sin(2.0 * math.pi * rng.uniform(0.7, 2.4) * t)
        audio = 0.38 * saw(root * mod, t) + 0.25 * square(root * 1.51, t) + rng.normal(0.0, 0.18, n)
        audio *= env(n, sr, 0.02, 0.18, 0.86)
        audio = board(audio, sr, [Distortion(drive_db=rng.uniform(4, 10)), LowpassFilter(cutoff_frequency_hz=rng.uniform(1800, 5200)), Chorus(rate_hz=1.2, depth=0.35, mix=0.24)])
        return stereoize(audio, rng, 0.42)

    if label == "fuzzy_lofi_synth":
        wobble = 1.0 + 0.012 * np.sin(2.0 * math.pi * rng.uniform(0.9, 1.8) * t)
        audio = 0.58 * saw(root * wobble, t) + 0.16 * square(root * 0.5, t) + rng.normal(0.0, 0.045, n)
        audio *= env(n, sr, 0.04, 0.22, 0.74)
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(950, 2400)), Distortion(drive_db=rng.uniform(4, 9)), Reverb(room_size=0.28, wet_level=0.08)])
        return stereoize(audio, rng, 0.25)

    if label == "synth_pluck_bell":
        partials = sine(root * 3.0, t) + 0.62 * sine(root * 5.07, t) + 0.21 * sine(root * 8.1, t)
        audio = partials * np.exp(-t * rng.uniform(4.5, 9.5))
        audio *= gate(t, rng.choice([3.0, 4.0, 6.0]), rng.uniform(0.28, 0.5))
        return stereoize(board(audio, sr, [HighpassFilter(cutoff_frequency_hz=280), Reverb(room_size=0.33, wet_level=0.12)]), rng, 0.22)

    if label == "arpeggio_sequence":
        notes = np.array([root, root * 1.25, root * 1.5, root * 2.0, root * 2.5])
        freq = notes[np.floor((t * rng.choice([6.0, 8.0, 10.0])) % len(notes)).astype(int)]
        audio = (0.45 * square(freq, t) + 0.35 * sine(freq * 2.0, t)) * gate(t, rng.choice([6.0, 8.0, 12.0]), rng.uniform(0.32, 0.55))
        return stereoize(board(audio, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(2200, 6200)), Reverb(room_size=0.28, wet_level=0.1)]), rng, 0.24)

    if label == "not_synth_drums":
        audio = np.zeros(n, dtype=np.float32)
        for hit in np.arange(0, duration, 0.25):
            start = int(hit * sr)
            length = min(int(0.035 * sr), n - start)
            if length <= 0:
                continue
            audio[start : start + length] += rng.normal(0.0, 0.9, length) * np.exp(-np.linspace(0, 8, length))
        audio = board(audio, sr, [HighpassFilter(cutoff_frequency_hz=1200), Bitcrush(bit_depth=7)])
        return stereoize(audio, rng, 0.18)

    if label == "not_synth_sample_loop":
        audio = 0.24 * saw(root, t) + 0.24 * sine(root * 1.5, t) + rng.normal(0.0, 0.11, n)
        audio *= gate(t, rng.choice([3.0, 4.0]), rng.uniform(0.55, 0.74))
        audio = board(audio, sr, [LowpassFilter(cutoff_frequency_hz=rng.uniform(700, 1800)), Reverb(room_size=0.28, wet_level=0.1)])
        return stereoize(audio, rng, 0.36)

    return stereoize(rng.normal(0.0, 0.1, n).astype(np.float32), rng)


def family_for(label: str) -> str:
    if label.startswith("not_synth"):
        return "not_synth"
    return "synth"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/synth_contrast_v1"))
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=711)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    index = 0
    for label, count in COUNTS.items():
        for source_index in range(count):
            audio = normalize(render(label, args.duration, args.sample_rate, rng))
            out_file = audio_dir / f"contrast_{index:06d}_{label}.wav"
            sf.write(out_file, audio.T if audio.ndim == 2 else audio, args.sample_rate)
            rows.append(
                {
                    "file": out_file.relative_to(args.out).as_posix(),
                    "label": label,
                    "family": family_for(label),
                    "base_label": f"contrast:{label}",
                    "duration": args.duration,
                    "source_index": source_index,
                }
            )
            index += 1
        print(f"{label}: {count}")

    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"clips: {len(rows)}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
