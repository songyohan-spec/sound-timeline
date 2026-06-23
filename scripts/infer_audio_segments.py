from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.features import extract_audio_features


def load_audio(path: Path, target_sr: int = 44_100) -> tuple[np.ndarray, int]:
    import librosa

    try:
        audio, sr = sf.read(path, always_2d=True)
    except Exception:
        loaded, sr = librosa.load(path, sr=None, mono=False)
        if loaded.ndim == 1:
            audio = loaded[:, None]
        else:
            audio = loaded.T
    if sr != target_sr:
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(librosa.resample(audio[:, ch].astype(np.float32), orig_sr=sr, target_sr=target_sr))
        min_len = min(len(ch) for ch in channels)
        audio = np.stack([ch[:min_len] for ch in channels], axis=1)
        sr = target_sr
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2].astype(np.float32), sr


def top_predictions(model, encoder, x, top_k: int) -> list[dict]:
    probs = model.predict_proba(x)[0]
    order = probs.argsort()[::-1][:top_k]
    labels = encoder.inverse_transform(order)
    return [{"label": str(label), "confidence": round(float(probs[index]), 4)} for label, index in zip(labels, order)]


def analyze_segment(bundle: dict, wav_path: Path, top_k: int) -> dict:
    features = extract_audio_features(wav_path)
    x = bundle["scaler"].transform([features])
    predictions = {}
    for target in bundle["targets"]:
        predictions[target] = top_predictions(bundle["models"][target], bundle["encoders"][target], x, top_k)
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("outputs/segments_report.json"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.audio.exists():
        raise SystemExit(f"Audio not found: {args.audio}")

    bundle = joblib.load(args.model)
    audio, sr = load_audio(args.audio)
    seg_len = int(args.segment_seconds * sr)
    hop_len = int(args.hop_seconds * sr)
    if len(audio) < seg_len:
        pad = seg_len - len(audio)
        audio = np.pad(audio, ((0, pad), (0, 0)))

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        start = 0
        index = 0
        while start + seg_len <= len(audio):
            end = start + seg_len
            segment = audio[start:end]
            wav_path = tmp_dir / f"segment_{index:04d}.wav"
            sf.write(wav_path, segment, sr)
            predictions = analyze_segment(bundle, wav_path, args.top_k)
            segments.append(
                {
                    "index": index,
                    "start": round(start / sr, 3),
                    "end": round(end / sr, 3),
                    "predictions": predictions,
                }
            )
            index += 1
            start += hop_len

    report = {
        "audio": str(args.audio),
        "model": str(args.model),
        "segment_seconds": args.segment_seconds,
        "hop_seconds": args.hop_seconds,
        "segments": segments,
        "caution": "Segment-level estimates are categorical sound-profile hints, not source separation or exact effect-chain recovery.",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
