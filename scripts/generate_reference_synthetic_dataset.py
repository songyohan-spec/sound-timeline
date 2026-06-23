from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Bitcrush, Chorus, Distortion, HighpassFilter, LowpassFilter, Pedalboard, Reverb


DEFAULT_LABELS = [
    "processed_lead_vocal",
    "hard_tuned_vocal",
    "pitched_vocal_chop",
    "breathy_vocal_pad",
    "stacked_harmony_vocal",
    "vocal_synth_hybrid",
    "lush_synth_pad",
    "syrupy_video_game_synth_melody",
    "bitcrushed_synth_lead",
    "noisy_wavetable_texture",
    "fuzzy_diy_synth_texture",
    "synth_flute_or_recorder_like_lead",
    "sub_bass",
    "distorted_808_bass",
    "pulsing_sidechain_bass",
    "glitch_percussion",
    "trap_drum_pattern",
    "filtered_guitar_loop",
    "washed_chorus_guitar",
    "distorted_guitar_texture",
    "filtered_sample_loop",
    "unknown_hybrid_loop",
]

GROUP_BY_LABEL = {
    "processed_lead_vocal": "vocals",
    "hard_tuned_vocal": "vocals",
    "pitched_vocal_chop": "vocals",
    "breathy_vocal_pad": "vocals",
    "stacked_harmony_vocal": "vocals",
    "vocal_synth_hybrid": "sampled_loop",
    "lush_synth_pad": "synth",
    "syrupy_video_game_synth_melody": "synth",
    "bitcrushed_synth_lead": "synth",
    "noisy_wavetable_texture": "synth",
    "fuzzy_diy_synth_texture": "synth",
    "synth_flute_or_recorder_like_lead": "synth",
    "sub_bass": "bass",
    "distorted_808_bass": "bass",
    "pulsing_sidechain_bass": "bass",
    "glitch_percussion": "drums",
    "trap_drum_pattern": "drums",
    "live_drum_layer": "drums",
    "filtered_guitar_loop": "guitar_strings",
    "washed_chorus_guitar": "guitar_strings",
    "distorted_guitar_texture": "guitar_strings",
    "filtered_sample_loop": "sampled_loop",
    "unknown_hybrid_loop": "sampled_loop",
}


def normalize(audio: np.ndarray, peak: float = 0.92) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def sine(freq: float | np.ndarray, t: np.ndarray) -> np.ndarray:
    if isinstance(freq, np.ndarray):
        phase = np.cumsum(freq) / len(t) * (t[-1] if len(t) else 1.0)
        return np.sin(2 * math.pi * phase)
    return np.sin(2 * math.pi * freq * t)


def saw(freq: float, t: np.ndarray) -> np.ndarray:
    return 2.0 * ((freq * t) % 1.0) - 1.0


def square(freq: float, t: np.ndarray) -> np.ndarray:
    return np.sign(np.sin(2 * math.pi * freq * t))


def adsr(length: int, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    env = np.ones(length, dtype=np.float32) * sustain
    a = min(length, int(attack * sr))
    d = min(max(0, length - a), int(decay * sr))
    r = min(length, int(release * sr))
    if a > 0:
        env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    if d > 0:
        env[a : a + d] = np.linspace(1.0, sustain, d, dtype=np.float32)
    if r > 0:
        env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def rhythm_gate(t: np.ndarray, rate: float, duty: float = 0.48) -> np.ndarray:
    return (((t * rate) % 1.0) < duty).astype(np.float32)


def sidechain(audio: np.ndarray, sr: int, bpm: float = 120.0, depth: float = 0.85) -> np.ndarray:
    t = np.arange(len(audio)) / sr
    beat = 60.0 / bpm
    phase = (t % beat) / beat
    recover = np.clip(phase / 0.7, 0.0, 1.0)
    curve = 1.0 - depth * (1.0 - recover) ** 2.2
    return normalize(audio * curve)


def widen(audio: np.ndarray, amount: float = 0.65) -> np.ndarray:
    delay = 13
    right = np.roll(audio, delay) * amount + audio * (1.0 - amount)
    return normalize(np.stack([audio, right], axis=0))


def mono_or_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio
    return np.stack([audio, audio], axis=0)


def board_process(audio: np.ndarray, sr: int, pedals: list) -> np.ndarray:
    rendered = Pedalboard(pedals)(audio.astype(np.float32), sr)
    return normalize(rendered)


def kick(t: np.ndarray, sr: int, at: float) -> np.ndarray:
    x = np.zeros_like(t)
    start = int(at * sr)
    dur = int(0.22 * sr)
    if start >= len(x):
        return x
    n = min(dur, len(x) - start)
    local = np.arange(n) / sr
    freq = 120.0 * np.exp(-local * 18.0) + 42.0
    phase = np.cumsum(freq) / sr
    x[start : start + n] += np.sin(2 * math.pi * phase) * np.exp(-local * 18.0)
    return x


def noise_hit(t: np.ndarray, sr: int, at: float, dur: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros_like(t)
    start = int(at * sr)
    n = min(int(dur * sr), max(0, len(x) - start))
    if n <= 0:
        return x
    local = np.linspace(0.0, 1.0, n, endpoint=False)
    x[start : start + n] = rng.normal(0.0, 1.0, n) * np.exp(-local * 9.0)
    return x


def vocal_tone(
    t: np.ndarray,
    fundamental: np.ndarray | float,
    rng: np.random.Generator,
    breath: float = 0.04,
    formant_shift: float = 1.0,
) -> np.ndarray:
    # A deliberately simple voiced source: harmonic stack plus broad vowel-like
    # resonances. It is not speech synthesis, but it gives the classifier more
    # vocal-specific spectral structure than a plain sine wave.
    audio = np.zeros_like(t, dtype=np.float32)
    harmonic_weights = [1.0, 0.52, 0.28, 0.16, 0.09, 0.05]
    for harmonic, weight in enumerate(harmonic_weights, start=1):
        audio += weight * sine(np.asarray(fundamental) * harmonic, t)
    audio += rng.normal(0.0, breath, len(t)).astype(np.float32)
    return normalize(audio)


def vowel_filter(audio: np.ndarray, sr: int, profile: str = "open") -> np.ndarray:
    if profile == "bright":
        pedals = [HighpassFilter(cutoff_frequency_hz=180), LowpassFilter(cutoff_frequency_hz=5_200)]
    elif profile == "dark":
        pedals = [HighpassFilter(cutoff_frequency_hz=120), LowpassFilter(cutoff_frequency_hz=2_200)]
    elif profile == "thin":
        pedals = [HighpassFilter(cutoff_frequency_hz=520), LowpassFilter(cutoff_frequency_hz=4_400)]
    else:
        pedals = [HighpassFilter(cutoff_frequency_hz=150), LowpassFilter(cutoff_frequency_hz=3_600)]
    return board_process(audio, sr, pedals)


def make_audio(label: str, duration: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    root = rng.choice([110.0, 130.81, 146.83, 164.81, 196.0, 220.0])

    if label == "sub_bass":
        audio = sine(rng.choice([43.65, 49.0, 55.0, 65.41]), t) * adsr(len(t), sr, 0.01, 0.08, 0.8, 0.08)
        return mono_or_stereo(normalize(audio))
    if label == "distorted_808_bass":
        audio = sine(48.0 + 10.0 * np.exp(-t * 4.0), t) * np.exp(-t * 0.45)
        audio = board_process(audio, sr, [Distortion(drive_db=rng.uniform(12, 24)), LowpassFilter(cutoff_frequency_hz=900)])
        return mono_or_stereo(audio)
    if label == "pulsing_sidechain_bass":
        audio = sine(55.0, t) + 0.22 * saw(110.0, t)
        audio = board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=650)])
        return mono_or_stereo(sidechain(audio, sr, depth=0.9))

    if label == "lush_synth_pad":
        audio = sum(saw(root * ratio, t + rng.uniform(0, 0.01)) for ratio in [1.0, 1.005, 0.995, 2.0]) / 4
        audio *= adsr(len(t), sr, 0.45, 0.3, 0.75, 0.5)
        audio = board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=2200), Chorus(rate_hz=0.8, depth=0.7, mix=0.55), Reverb(room_size=0.82, wet_level=0.38, dry_level=0.72)])
        return widen(audio, 0.78)
    if label == "syrupy_video_game_synth_melody":
        freqs = np.array([root, root * 1.25, root * 1.5, root * 2.0])
        step = np.floor((t * 8.0) % len(freqs)).astype(int)
        audio = square(freqs[step], t) * adsr(len(t), sr, 0.005, 0.05, 0.65, 0.05)
        audio = board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=4200), Chorus(rate_hz=1.2, depth=0.35, mix=0.25)])
        return mono_or_stereo(audio)
    if label == "bitcrushed_synth_lead":
        audio = saw(root * 2.0, t) * rhythm_gate(t, rng.choice([4.0, 8.0]), 0.55)
        audio = board_process(audio, sr, [Distortion(drive_db=14), Bitcrush(bit_depth=rng.choice([5, 6, 7])), HighpassFilter(cutoff_frequency_hz=250)])
        return mono_or_stereo(audio)
    if label == "noisy_wavetable_texture":
        audio = 0.45 * saw(root, t) + 0.35 * square(root * 0.5, t) + rng.normal(0.0, 0.22, len(t))
        audio = board_process(audio, sr, [Distortion(drive_db=10), Chorus(rate_hz=1.8, depth=0.55, mix=0.35), Reverb(room_size=0.55, wet_level=0.25)])
        return widen(audio, 0.6)
    if label == "fuzzy_diy_synth_texture":
        wobble = 1.0 + 0.012 * np.sin(2 * math.pi * 1.2 * t)
        audio = saw(root * wobble.mean(), t) + 0.12 * rng.normal(0.0, 1.0, len(t))
        audio = board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=1800), Distortion(drive_db=7), Reverb(room_size=0.45, wet_level=0.18)])
        return mono_or_stereo(audio)
    if label == "synth_flute_or_recorder_like_lead":
        vibrato = 4.0 * np.sin(2 * math.pi * 5.5 * t)
        audio = sine(root * 2 + vibrato, t) + 0.12 * sine(root * 4 + vibrato, t)
        audio *= adsr(len(t), sr, 0.08, 0.16, 0.85, 0.2)
        return mono_or_stereo(board_process(audio, sr, [HighpassFilter(cutoff_frequency_hz=180), Reverb(room_size=0.35, wet_level=0.13)]))

    if label == "processed_lead_vocal":
        vibrato = 5.0 * np.sin(2 * math.pi * rng.uniform(4.8, 6.4) * t)
        phrase = root * rng.choice([1.5, 2.0, 2.25]) + vibrato
        audio = vocal_tone(t, phrase, rng, breath=0.055)
        audio *= adsr(len(t), sr, 0.018, 0.1, 0.74, 0.16)
        audio = vowel_filter(audio, sr, rng.choice(["open", "bright", "dark"]))
        return mono_or_stereo(board_process(audio, sr, [Distortion(drive_db=rng.uniform(3, 7)), Chorus(rate_hz=1.0, depth=0.22, mix=0.18), Reverb(room_size=0.30, wet_level=0.14)]))
    if label == "hard_tuned_vocal":
        notes = np.array([root * 1.5, root * 2.0, root * 2.25, root * 3.0, root * 3.375])
        idx = np.floor((t * rng.choice([4.0, 5.0, 6.0])) % len(notes)).astype(int)
        stepped = notes[idx]
        audio = vocal_tone(t, stepped, rng, breath=0.025)
        audio *= adsr(len(t), sr, 0.004, 0.035, 0.82, 0.04)
        audio = vowel_filter(audio, sr, "bright")
        return mono_or_stereo(board_process(audio, sr, [Distortion(drive_db=3.5), Reverb(room_size=0.34, wet_level=0.13)]))
    if label == "pitched_vocal_chop":
        notes = np.array([root * 1.5, root * 2.0, root * 2.52, root * 3.0, root * 4.0])
        idx = np.floor((t * rng.choice([6.0, 8.0, 10.0])) % len(notes)).astype(int)
        audio = vocal_tone(t, notes[idx], rng, breath=0.035)
        audio *= rhythm_gate(t, rng.choice([6.0, 8.0, 12.0]), rng.uniform(0.26, 0.44))
        audio = vowel_filter(audio, sr, rng.choice(["thin", "bright"]))
        return mono_or_stereo(board_process(audio, sr, [HighpassFilter(cutoff_frequency_hz=260), Reverb(room_size=0.52, wet_level=0.16)]))
    if label == "breathy_vocal_pad":
        slow = root * rng.choice([1.0, 1.25, 1.5])
        audio = vocal_tone(t, slow + 2.0 * np.sin(2 * math.pi * 0.35 * t), rng, breath=0.12)
        audio *= adsr(len(t), sr, 0.55, 0.25, 0.68, 0.75)
        audio = vowel_filter(audio, sr, "dark")
        return widen(board_process(audio, sr, [Chorus(rate_hz=0.55, depth=0.72, mix=0.5), Reverb(room_size=0.92, wet_level=0.55, dry_level=0.55)]), 0.76)
    if label == "stacked_harmony_vocal":
        voices = []
        for ratio in [1.5, 2.0, 2.5, 3.0]:
            detune = 1.0 + rng.uniform(-0.004, 0.004)
            voices.append(vocal_tone(t, root * ratio * detune, rng, breath=0.025))
        audio = sum(voices) / len(voices)
        audio *= adsr(len(t), sr, 0.07, 0.12, 0.78, 0.25)
        audio = vowel_filter(audio, sr, "open")
        return widen(board_process(audio, sr, [Chorus(rate_hz=1.05, depth=0.5, mix=0.42), Reverb(room_size=0.55, wet_level=0.24)]), 0.6)
    if label == "vocal_synth_hybrid":
        voiced = vocal_tone(t, root * rng.choice([1.5, 2.0, 2.25]), rng, breath=0.035)
        synth_edge = 0.34 * saw(root * 2.0, t) + 0.18 * square(root * 1.0, t)
        audio = 0.62 * voiced + synth_edge
        audio *= rhythm_gate(t, rng.choice([3.0, 4.0, 6.0]), 0.62)
        audio = vowel_filter(audio, sr, "bright")
        return widen(board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=3_200), Chorus(rate_hz=1.5, depth=0.42, mix=0.38), Distortion(drive_db=3)]), 0.48)

    if label == "filtered_guitar_loop":
        audio = sine(root * 1.5, t) * np.exp(-5.0 * (t % 0.5)) + 0.3 * sine(root * 3, t) * np.exp(-6.0 * (t % 0.5))
        return mono_or_stereo(board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=1700)]))
    if label == "washed_chorus_guitar":
        audio = sine(root * 1.5, t) * np.exp(-3.0 * (t % 0.5))
        return widen(board_process(audio, sr, [Chorus(rate_hz=1.2, depth=0.8, mix=0.55), Reverb(room_size=0.86, wet_level=0.45)]), 0.72)
    if label == "distorted_guitar_texture":
        audio = sine(root, t) * np.exp(-2.8 * (t % 0.25)) + 0.25 * saw(root * 0.5, t)
        return mono_or_stereo(board_process(audio, sr, [Distortion(drive_db=20), LowpassFilter(cutoff_frequency_hz=2800)]))

    if label == "glitch_percussion":
        audio = np.zeros_like(t)
        for at in np.arange(0, duration, 0.125):
            if rng.random() < 0.55:
                audio += noise_hit(t, sr, float(at), rng.uniform(0.015, 0.055), rng)
        return mono_or_stereo(board_process(audio, sr, [HighpassFilter(cutoff_frequency_hz=1800), Bitcrush(bit_depth=6)]))
    if label == "trap_drum_pattern":
        audio = np.zeros_like(t)
        for at in np.arange(0, duration, 1.0):
            audio += kick(t, sr, float(at))
        for at in np.arange(0.5, duration, 1.0):
            audio += noise_hit(t, sr, float(at), 0.12, rng) * 0.55
        for at in np.arange(0, duration, 0.125):
            audio += noise_hit(t, sr, float(at), 0.025, rng) * 0.18
        return mono_or_stereo(board_process(audio, sr, [HighpassFilter(cutoff_frequency_hz=35)]))

    if label == "filtered_sample_loop":
        audio = 0.45 * saw(root, t) + 0.32 * sine(root * 1.5, t) + 0.18 * rng.normal(0.0, 1.0, len(t))
        audio *= rhythm_gate(t, 4.0, 0.7)
        return widen(board_process(audio, sr, [LowpassFilter(cutoff_frequency_hz=1450), Reverb(room_size=0.35, wet_level=0.14)]), 0.35)
    if label == "unknown_hybrid_loop":
        audio = 0.35 * saw(root, t) + 0.35 * sine(root * 2.2, t) + 0.18 * rng.normal(0.0, 1.0, len(t))
        audio *= rhythm_gate(t, rng.choice([3.0, 4.0, 6.0]), 0.55)
        return widen(board_process(audio, sr, [Chorus(rate_hz=1.4, depth=0.5, mix=0.3), Distortion(drive_db=5)]), 0.45)

    audio = rng.normal(0.0, 0.15, len(t))
    return mono_or_stereo(normalize(audio))


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/reference_element_dataset_synth_v1"))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--count-per-label", type=int, default=90)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    idx = 0
    for label in labels:
        group = GROUP_BY_LABEL.get(label, "unknown")
        for label_idx in range(args.count_per_label):
            audio = make_audio(label, args.duration, args.sample_rate, rng)
            out_file = audio_dir / f"refsyn_{idx:06d}_{label}.wav"
            sf.write(out_file, audio.T if audio.ndim == 2 else audio, args.sample_rate)
            rows.append(
                {
                    "file": out_file.relative_to(args.out).as_posix(),
                    "duration": args.duration,
                    "label": label,
                    "group": group,
                    "priority": "synthetic_reference",
                    "source_file": f"generated:{label}",
                    "source_index": label_idx,
                }
            )
            idx += 1
        summary.append({"label": label, "group": group, "clips": args.count_per_label})

    write_jsonl(rows, args.out / "metadata.jsonl")
    with (args.out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "group", "clips"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"clips: {len(rows)}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")
    print(f"summary: {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
