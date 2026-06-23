from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_mono(path: Path, target_sr: int = 22_050) -> tuple[np.ndarray, int]:
    import librosa

    audio, sr = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return mono, sr


def extract_audio_features(path: Path) -> np.ndarray:
    import librosa

    y, sr = load_mono(path)
    if len(y) < sr // 4:
        y = np.pad(y, (0, sr // 4 - len(y)))

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    band_features = extract_band_energy_features(y, sr)
    band_motion_features = extract_band_motion_features(y, sr)
    stereo_features = extract_stereo_features(path)
    envelope_features = extract_envelope_features(y, sr)

    stats = [
        np.mean(rms),
        np.std(rms),
        np.mean(zcr),
        np.std(zcr),
        np.mean(centroid),
        np.std(centroid),
        np.mean(bandwidth),
        np.std(bandwidth),
        np.mean(rolloff),
        np.std(rolloff),
        np.mean(flatness),
        np.std(flatness),
    ]

    mfcc_stats = []
    for row in mfcc:
        mfcc_stats.extend([np.mean(row), np.std(row)])

    return np.asarray(stats + mfcc_stats + band_features + band_motion_features + stereo_features + envelope_features, dtype=np.float32)


def extract_band_energy_features(y: np.ndarray, sr: int) -> list[float]:
    import librosa

    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    total = np.sum(stft, axis=0) + 1e-8

    bands = [
        (20.0, 120.0),
        (120.0, 400.0),
        (400.0, 1_500.0),
        (1_500.0, 4_000.0),
        (4_000.0, 10_000.0),
    ]

    features: list[float] = []
    band_means = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            ratio = np.zeros_like(total)
        else:
            ratio = np.sum(stft[mask], axis=0) / total
        band_means.append(float(np.mean(ratio)))
        features.extend([float(np.mean(ratio)), float(np.std(ratio))])

    low_energy = band_means[0] + band_means[1]
    high_energy = band_means[3] + band_means[4]
    features.append(float(high_energy / (low_energy + 1e-8)))
    features.append(float((band_means[4] + 1e-8) / (band_means[2] + 1e-8)))
    return features


def extract_band_motion_features(y: np.ndarray, sr: int) -> list[float]:
    import librosa

    n_fft = 2048
    hop_length = 512
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total = np.sum(stft, axis=0) + 1e-8

    low_mask = freqs < 700.0
    mid_mask = (freqs >= 700.0) & (freqs < 2_500.0)
    high_mask = freqs >= 2_500.0

    def ratio(mask: np.ndarray) -> np.ndarray:
        if not np.any(mask):
            return np.zeros_like(total)
        return np.sum(stft[mask], axis=0) / total

    low = ratio(low_mask)
    mid = ratio(mid_mask)
    high = ratio(high_mask)
    brightness = high / (low + mid + 1e-8)
    low_dominance = low / (mid + high + 1e-8)

    def slope(values: np.ndarray) -> float:
        if len(values) < 3:
            return 0.0
        x = np.linspace(-1.0, 1.0, len(values), dtype=np.float32)
        return float(np.polyfit(x, values.astype(np.float32), deg=1)[0])

    thirds = np.array_split(np.arange(len(brightness)), 3)
    bright_thirds = [float(np.mean(brightness[idx])) if len(idx) else 0.0 for idx in thirds]
    low_thirds = [float(np.mean(low_dominance[idx])) if len(idx) else 0.0 for idx in thirds]

    return [
        slope(brightness),
        slope(low_dominance),
        bright_thirds[-1] - bright_thirds[0],
        low_thirds[-1] - low_thirds[0],
        float(np.std(brightness)),
        float(np.std(low_dominance)),
    ]


def extract_stereo_features(path: Path) -> list[float]:
    audio, _ = sf.read(path, always_2d=True)
    if audio.shape[1] < 2:
        return [0.0] * 8

    left = audio[:, 0].astype(np.float32)
    right = audio[:, 1].astype(np.float32)
    if len(left) < 2:
        return [0.0] * 8

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    left_rms = float(np.sqrt(np.mean(left**2) + 1e-8))
    right_rms = float(np.sqrt(np.mean(right**2) + 1e-8))
    mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-8))
    side_rms = float(np.sqrt(np.mean(side**2) + 1e-8))
    corr = float(np.corrcoef(left, right)[0, 1]) if np.std(left) > 1e-8 and np.std(right) > 1e-8 else 1.0

    diff = left - right
    diff_rms = float(np.sqrt(np.mean(diff**2) + 1e-8))
    balance = float(abs(left_rms - right_rms) / (left_rms + right_rms + 1e-8))

    return [
        corr,
        side_rms / (mid_rms + 1e-8),
        diff_rms / (left_rms + right_rms + 1e-8),
        balance,
        left_rms,
        right_rms,
        mid_rms,
        side_rms,
    ]


def extract_envelope_features(y: np.ndarray, sr: int) -> list[float]:
    import librosa

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0].astype(np.float32)
    if len(rms) < 8 or float(np.max(rms)) < 1e-8:
        return [0.0] * 10

    rms = rms / (float(np.max(rms)) + 1e-8)
    centered = rms - float(np.mean(rms))

    # Autocorrelation of the loudness envelope: sidechain pumping often creates
    # periodic dips/recovery curves that are visible here even when spectrum
    # averages look ordinary.
    corr = np.correlate(centered, centered, mode="full")[len(centered) - 1 :]
    corr = corr / (float(corr[0]) + 1e-8)

    frame_rate = sr / hop_length
    min_lag = max(1, int(frame_rate * 0.20))
    max_lag = min(len(corr) - 1, int(frame_rate * 1.10))
    if max_lag <= min_lag:
        periodic_peak = 0.0
        periodic_lag = 0.0
    else:
        lag_window = corr[min_lag:max_lag]
        best = int(np.argmax(lag_window)) + min_lag
        periodic_peak = float(corr[best])
        periodic_lag = float(best / frame_rate)

    diff = np.diff(rms)
    negative_dips = diff[diff < 0]
    positive_recovery = diff[diff > 0]

    # Frequency-domain summary of the envelope. This gives the model an easy
    # handle on slow amplitude modulation.
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(len(centered), d=1.0 / frame_rate)
    band = (freqs >= 0.8) & (freqs <= 8.0)
    if np.any(band):
        modulation_strength = float(np.max(spectrum[band]) / (np.sum(spectrum) + 1e-8))
        modulation_freq = float(freqs[band][np.argmax(spectrum[band])])
    else:
        modulation_strength = 0.0
        modulation_freq = 0.0

    return [
        float(np.mean(rms)),
        float(np.std(rms)),
        float(np.max(rms) - np.min(rms)),
        float(np.percentile(rms, 90) - np.percentile(rms, 10)),
        periodic_peak,
        periodic_lag,
        float(np.mean(np.abs(diff))),
        float(np.mean(np.abs(negative_dips))) if len(negative_dips) else 0.0,
        modulation_strength,
        modulation_freq,
    ]
