from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def sigmoid01(value: float, center: float, scale: float) -> float:
    return clamp01(1.0 / (1.0 + np.exp(-(value - center) / scale)))


def log_band_features(power: np.ndarray, freqs: np.ndarray, axis: int) -> dict[str, float]:
    edges = np.geomspace(40.0, 11_000.0, 13)
    features: dict[str, float] = {}
    if axis == 1:
        total_by_frame = np.sum(power, axis=1) + 1e-8
        for idx in range(12):
            mask = (freqs >= edges[idx]) & (freqs < edges[idx + 1])
            if not np.any(mask):
                ratio = np.zeros(power.shape[0], dtype=np.float32)
            else:
                ratio = np.sum(power[:, mask], axis=1) / total_by_frame
            features[f"band_{idx:02d}_mean"] = float(np.mean(ratio))
            features[f"band_{idx:02d}_std"] = float(np.std(ratio))
    else:
        total_by_frame = np.sum(power, axis=0) + 1e-8
        for idx in range(12):
            mask = (freqs >= edges[idx]) & (freqs < edges[idx + 1])
            if not np.any(mask):
                ratio = np.zeros(power.shape[1], dtype=np.float32)
            else:
                ratio = np.sum(power[mask, :], axis=0) / total_by_frame
            features[f"band_{idx:02d}_mean"] = float(np.mean(ratio))
            features[f"band_{idx:02d}_std"] = float(np.std(ratio))
    return features


def load_audio(path: Path) -> tuple[np.ndarray, int, np.ndarray]:
    audio, sr = sf.read(path, always_2d=True)
    stereo = audio.astype(np.float32)
    mono = stereo.mean(axis=1).astype(np.float32)
    return mono, sr, stereo


def audio_stats_fast(path: Path) -> dict[str, float]:
    y, sr, stereo = load_audio(path)
    if len(y) < 1024:
        y = np.pad(y, (0, 1024 - len(y)))

    # Downsample by slicing for speed. These are heuristic cues, not
    # production-grade measurements.
    if sr > 24_000:
        step = max(1, int(sr // 24_000))
        y = y[::step]
        stereo = stereo[::step]
        sr = int(sr / step)

    frame_size = min(2048, max(512, len(y) // 4))
    hop = max(256, frame_size // 2)
    if len(y) < frame_size:
        y = np.pad(y, (0, frame_size - len(y)))
    starts = range(0, max(1, len(y) - frame_size + 1), hop)
    frames = np.stack([y[start : start + frame_size] for start in starts]).astype(np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)) + 1e-8
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    power = spectrum**2
    total = np.sum(power, axis=1) + 1e-8
    centroid = np.sum(power * freqs[None, :], axis=1) / total
    bandwidth = np.sqrt(np.sum(power * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / total)
    flatness = np.exp(np.mean(np.log(spectrum), axis=1)) / (np.mean(spectrum, axis=1) + 1e-8)
    cumulative = np.cumsum(power, axis=1)
    rolloff_bins = np.argmax(cumulative >= (0.85 * total[:, None]), axis=1)
    rolloff = freqs[rolloff_bins]
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-8)
    mean_power = np.mean(power, axis=0)
    total_power_mean = float(np.sum(mean_power) + 1e-8)
    low_energy = float(np.sum(mean_power[(freqs >= 20) & (freqs < 160)]) / total_power_mean)
    low_mid_energy = float(np.sum(mean_power[(freqs >= 160) & (freqs < 600)]) / total_power_mean)
    mid_energy = float(np.sum(mean_power[(freqs >= 600) & (freqs < 2_500)]) / total_power_mean)
    high_energy = float(np.sum(mean_power[(freqs >= 2_500) & (freqs < 8_000)]) / total_power_mean)
    air_energy = float(np.sum(mean_power[freqs >= 8_000]) / total_power_mean)
    flux = np.diff(np.log1p(spectrum), axis=0)
    spectral_flux = float(np.mean(np.sqrt(np.mean(flux**2, axis=1)))) if len(flux) else 0.0

    rms_norm = rms / (float(np.max(rms)) + 1e-8)
    centered = rms_norm - float(np.mean(rms_norm))
    env_spec = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), d=512 / sr)
    motion_band = (freqs >= 0.7) & (freqs <= 8.0)
    if np.any(motion_band):
        motion_strength = float(np.max(env_spec[motion_band]) / (np.sum(env_spec) + 1e-8))
        motion_freq = float(freqs[motion_band][np.argmax(env_spec[motion_band])])
    else:
        motion_strength = 0.0
        motion_freq = 0.0

    if stereo.shape[1] >= 2:
        left = stereo[:, 0]
        right = stereo[:, 1]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        width = float(np.sqrt(np.mean(side**2) + 1e-8) / (np.sqrt(np.mean(mid**2) + 1e-8) + 1e-8))
    else:
        width = 0.0

    features = {
        "centroid": float(np.mean(centroid)),
        "bandwidth": float(np.mean(bandwidth)),
        "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(zcr)),
        "rolloff": float(np.mean(rolloff)),
        "low_energy": low_energy,
        "low_mid_energy": low_mid_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "air_energy": air_energy,
        "spectral_flux": spectral_flux,
        "crest_factor": float((np.max(np.abs(y)) + 1e-8) / (np.sqrt(np.mean(y**2) + 1e-8) + 1e-8)),
        "rms_std": float(np.std(rms_norm)),
        "rms_range": float(np.max(rms_norm) - np.min(rms_norm)),
        "motion_strength": motion_strength,
        "motion_freq": motion_freq,
        "width": width,
    }
    features.update(log_band_features(power, freqs, axis=1))
    return features


def audio_stats_librosa(path: Path) -> dict[str, float]:
    from scipy.signal import stft

    y, sr, stereo = load_audio(path)
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)))

    target_sr = 22_050
    if sr != target_sr:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(sr, target_sr))
        y = resample_poly(y, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr

    n_fft = 2048
    hop_length = 512
    _, _, zxx = stft(
        y,
        fs=sr,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop_length,
        nfft=n_fft,
        boundary=None,
        padded=False,
    )
    mag = np.abs(zxx).astype(np.float32) + 1e-8
    power = mag**2
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr).astype(np.float32)
    total = np.sum(power, axis=0) + 1e-8

    centroid = np.sum(power * freqs[:, None], axis=0) / total
    bandwidth = np.sqrt(np.sum(power * (freqs[:, None] - centroid[None, :]) ** 2, axis=0) / total)
    flatness = np.exp(np.mean(np.log(mag), axis=0)) / (np.mean(mag, axis=0) + 1e-8)
    cumulative = np.cumsum(power, axis=0)
    rolloff_bins = np.argmax(cumulative >= (0.85 * total[None, :]), axis=0)
    rolloff = freqs[rolloff_bins]
    frame_count = max(1, 1 + (len(y) - n_fft) // hop_length)
    frames = np.stack([y[i * hop_length : i * hop_length + n_fft] for i in range(frame_count)]).astype(np.float32)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames), axis=1)), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-8).astype(np.float32)
    mean_power = np.mean(power, axis=1)
    total_power_mean = float(np.sum(mean_power) + 1e-8)
    low_energy = float(np.sum(mean_power[(freqs >= 20) & (freqs < 160)]) / total_power_mean)
    low_mid_energy = float(np.sum(mean_power[(freqs >= 160) & (freqs < 600)]) / total_power_mean)
    mid_energy = float(np.sum(mean_power[(freqs >= 600) & (freqs < 2_500)]) / total_power_mean)
    high_energy = float(np.sum(mean_power[(freqs >= 2_500) & (freqs < 8_000)]) / total_power_mean)
    air_energy = float(np.sum(mean_power[freqs >= 8_000]) / total_power_mean)
    flux = np.diff(np.log1p(mag), axis=1)
    spectral_flux = float(np.mean(np.sqrt(np.mean(flux**2, axis=0)))) if flux.size else 0.0

    rms_norm = rms / (float(np.max(rms)) + 1e-8)
    centered = rms_norm - float(np.mean(rms_norm))
    env_spec = np.abs(np.fft.rfft(centered))
    env_freqs = np.fft.rfftfreq(len(centered), d=hop_length / sr)
    motion_band = (env_freqs >= 0.7) & (env_freqs <= 8.0)
    if np.any(motion_band):
        motion_strength = float(np.max(env_spec[motion_band]) / (np.sum(env_spec) + 1e-8))
        motion_freq = float(env_freqs[motion_band][np.argmax(env_spec[motion_band])])
    else:
        motion_strength = 0.0
        motion_freq = 0.0

    if stereo.shape[1] >= 2:
        left = stereo[:, 0].astype(np.float32)
        right = stereo[:, 1].astype(np.float32)
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right)
        width = float(np.sqrt(np.mean(side**2) + 1e-8) / (np.sqrt(np.mean(mid**2) + 1e-8) + 1e-8))
    else:
        width = 0.0

    features = {
        "centroid": float(np.mean(centroid)),
        "bandwidth": float(np.mean(bandwidth)),
        "flatness": float(np.mean(flatness)),
        "zcr": float(np.mean(zcr)),
        "rolloff": float(np.mean(rolloff)),
        "low_energy": low_energy,
        "low_mid_energy": low_mid_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "air_energy": air_energy,
        "spectral_flux": spectral_flux,
        "crest_factor": float((np.max(np.abs(y)) + 1e-8) / (np.sqrt(np.mean(y**2) + 1e-8) + 1e-8)),
        "rms_std": float(np.std(rms_norm)),
        "rms_range": float(np.max(rms_norm) - np.min(rms_norm)),
        "motion_strength": motion_strength,
        "motion_freq": motion_freq,
        "width": width,
    }
    features.update(log_band_features(power, freqs, axis=0))
    return features


def audio_stats(path: Path, quality: str = "librosa") -> dict[str, float]:
    if quality == "fast":
        return audio_stats_fast(path)
    return audio_stats_librosa(path)


def item(family: str, label: str, score: float, prompt: str) -> dict:
    return {
        "family": family,
        "label": label,
        "score": round(clamp01(score), 8),
        "prompt": prompt,
    }


class DSPPaletteScorer:
    model_name = "dsp-heuristic-palette"

    def __init__(self, palette_path: Path | None = None, quality: str = "librosa") -> None:
        self.palette_path = palette_path
        self.quality = quality
        self.model_name = f"dsp-heuristic-palette-{quality}"

    def score(self, audio_path: Path) -> dict:
        s = audio_stats(audio_path, quality=self.quality)
        brightness = clamp01(s["centroid"] / 5000.0)
        noisiness = clamp01((s["flatness"] * 18.0) + (s["zcr"] * 2.0))
        motion = clamp01(s["motion_strength"] * 8.0 + s["rms_std"] * 0.6)
        lowpass_like = clamp01(1.0 - brightness)
        wide = clamp01(s["width"] * 1.8)

        items = [
            item("hybrid_sampled", "filtered_sample_loop", 0.30 + 0.45 * lowpass_like + 0.20 * motion, "DSP: muted/filtered loop-like spectrum with motion"),
            item("hybrid_sampled", "resampled_pop_texture", 0.20 + 0.35 * motion + 0.25 * noisiness, "DSP: processed sample-like texture"),
            item("hybrid_sampled", "vocal_synth_hybrid", 0.20 + 0.25 * (1.0 - noisiness) + 0.15 * brightness, "DSP: mid-bright tonal hybrid texture"),
            item("vocal_derived", "pitched_vocal_chop", 0.12 + 0.28 * motion + 0.14 * brightness, "DSP: rhythmic melodic chopped contour"),
            item("synth_derived", "bitcrushed_synth_lead", 0.08 + 0.45 * noisiness + 0.20 * brightness, "DSP: bright/noisy synthetic lead texture"),
            item("guitar_derived", "filtered_guitar_loop", 0.15 + 0.32 * lowpass_like + 0.12 * (1.0 - noisiness), "DSP: filtered plucked loop candidate"),
            item("guitar_derived", "chorus_arpeggio_loop", 0.08 + 0.25 * wide + 0.10 * (1.0 - noisiness), "DSP: wide clean picked loop candidate"),
            item("fx_texture", "digital_glitch_fill", 0.08 + 0.35 * noisiness + 0.10 * s["rms_range"], "DSP: noisy transient/digital texture"),
            item("fx_texture", "reverse_cymbal_swell", 0.06 + 0.22 * brightness + 0.08 * s["rms_range"], "DSP: bright transition-like texture"),
            item("processing_space", "rhythmic_pulse_motion", 0.12 + 0.70 * motion, "DSP: amplitude envelope pulse"),
            item("processing_space", "sidechain_pumping", 0.08 + 0.55 * motion + 0.08 * lowpass_like, "DSP: ducking-like amplitude motion"),
            item("processing_space", "gated_stutter_motion", 0.04 + 0.40 * motion + 0.18 * noisiness, "DSP: gated/chopped motion"),
            item("processing_space", "dry_close", 0.35 * (1.0 - wide) * (1.0 - motion), "DSP: narrow and low-motion"),
            item("processing_space", "wide_chorus_widening", 0.10 + 0.55 * wide, "DSP: strong side/mid width"),
        ]
        items = sorted(items, key=lambda row: row["score"], reverse=True)

        families: dict[str, list[dict]] = {}
        for row in items:
            families.setdefault(row["family"], []).append({k: row[k] for k in ["label", "score", "prompt"]})

        return {
            "audio": str(audio_path),
            "model": self.model_name,
            "stats": s,
            "families": families,
            "top_overall": items[:20],
            "caution": "DSP fallback scores are heuristic cues, not semantic CLAP probabilities or ground truth.",
        }
