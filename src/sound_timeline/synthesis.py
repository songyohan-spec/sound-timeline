from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pedalboard import Bitcrush, Chorus, Distortion, Gain, HighpassFilter, LowpassFilter, Pedalboard, Reverb


SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class SourceSpec:
    source: str
    role: str
    melody: str
    articulation: str


def normalize(audio: np.ndarray, peak: float = 0.9) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio / max_abs * peak).astype(np.float32)


def make_source(spec: SourceSpec, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)

    if spec.source == "bass":
        freq = 55.0 if spec.melody != "motif_repeated" else 55.0 + 8.0 * np.sin(2 * math.pi * 0.5 * t)
        audio = np.sin(2 * math.pi * freq * t)
    elif spec.source == "synth":
        freq = 220.0 if spec.melody != "motif_repeated" else 220.0 + 32.0 * np.sign(np.sin(2 * math.pi * 1.0 * t))
        phase = (freq * t) % 1.0
        audio = 2.0 * phase - 1.0
    elif spec.source == "guitar_like":
        carrier = np.sin(2 * math.pi * 196.0 * t)
        pluck = np.exp(-4.0 * (t % 0.5))
        audio = carrier * pluck
    elif spec.source == "vocal_like":
        vibrato = 4.0 * np.sin(2 * math.pi * 5.5 * t)
        audio = np.sin(2 * math.pi * (260.0 + vibrato) * t)
        audio += 0.35 * np.sin(2 * math.pi * (520.0 + vibrato) * t)
    elif spec.source == "noise_fx":
        audio = np.random.default_rng().normal(0.0, 0.35, size=t.shape)
    else:
        audio = np.zeros_like(t)

    envelope = np.ones_like(t)
    if spec.articulation == "swelling":
        envelope = np.linspace(0.0, 1.0, len(t)) ** 1.8
    elif spec.articulation == "plucked":
        envelope = np.exp(-3.0 * (t % 0.5))
    elif spec.articulation == "chopped":
        envelope = (np.sin(2 * math.pi * 8.0 * t) > 0).astype(float)
    elif spec.articulation == "pulsing":
        envelope = 0.5 + 0.5 * np.sin(2 * math.pi * 2.0 * t)

    return normalize(audio * envelope)


def apply_effects(audio: np.ndarray, effects: list[str], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    board = Pedalboard()

    if "lowpass_filter" in effects:
        board.append(LowpassFilter(cutoff_frequency_hz=1_800.0))
    if "highpass_filter" in effects:
        board.append(HighpassFilter(cutoff_frequency_hz=700.0))
    if "distortion" in effects:
        board.append(Distortion(drive_db=18.0))
    if "bitcrush" in effects:
        board.append(Bitcrush(bit_depth=8))
    if "chorus" in effects:
        board.append(Chorus(rate_hz=1.5, depth=0.6, mix=0.45))
    if "reverb" in effects:
        board.append(Reverb(room_size=0.75, damping=0.35, wet_level=0.35, dry_level=0.75))
    if "delay" in effects:
        delayed = np.zeros_like(audio)
        offset = int(sample_rate * 0.22)
        if offset < len(audio):
            delayed[offset:] = audio[:-offset] * 0.38
        audio = normalize(audio + delayed)

    board.append(Gain(gain_db=-2.0))
    return normalize(board(audio.astype(np.float32), sample_rate))


def apply_lowpass_sweep(audio: np.ndarray, start_hz: float = 500.0, end_hz: float = 7_500.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    chunks = np.array_split(audio, 24)
    rendered = []
    for idx, chunk in enumerate(chunks):
        alpha = idx / max(1, len(chunks) - 1)
        cutoff = start_hz + (end_hz - start_hz) * alpha
        rendered.append(Pedalboard([LowpassFilter(cutoff_frequency_hz=cutoff)])(chunk.astype(np.float32), sample_rate))
    return normalize(np.concatenate(rendered))


def apply_highpass_rise(audio: np.ndarray, start_hz: float = 80.0, end_hz: float = 2_800.0, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    chunks = np.array_split(audio, 24)
    rendered = []
    for idx, chunk in enumerate(chunks):
        alpha = idx / max(1, len(chunks) - 1)
        cutoff = start_hz + (end_hz - start_hz) * alpha
        rendered.append(Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff)])(chunk.astype(np.float32), sample_rate))
    return normalize(np.concatenate(rendered))


def apply_sidechain_pumping(audio: np.ndarray, bpm: float = 120.0, depth: float = 0.86, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(len(audio)) / sample_rate
    beat = 60.0 / bpm
    phase = (t % beat) / beat
    recovery = np.clip(phase / 0.72, 0.0, 1.0)
    curve = 1.0 - depth * (1.0 - recovery) ** 2.4
    return normalize(audio * curve)


def widen_mono(audio: np.ndarray, amount: float = 0.65) -> np.ndarray:
    delay = 19
    right = np.roll(audio, delay) * amount + audio * (1.0 - amount)
    stereo = np.stack([audio, right], axis=0)
    return normalize(stereo)


def to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio
    return np.stack([audio, audio], axis=0)


def db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))
