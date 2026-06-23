from __future__ import annotations

import argparse
import html
import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


def load_audio(path: Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
        stereo = audio.astype(np.float32)
    except Exception:
        import librosa

        loaded, sr = librosa.load(path, sr=sample_rate, mono=False)
        if loaded.ndim == 1:
            stereo = np.stack([loaded, loaded], axis=1).astype(np.float32)
        else:
            stereo = loaded.T.astype(np.float32)
    if sample_rate and sr != sample_rate:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(sr, sample_rate))
        stereo = resample_poly(stereo, sample_rate // gcd, sr // gcd, axis=0).astype(np.float32)
        sr = sample_rate
    return stereo, sr


def top_probs(model_bundle: dict, features: dict[str, float], target: str, top_k: int) -> list[dict]:
    model = model_bundle["models"][target]
    encoder = model_bundle["encoders"][target]
    cols = model_bundle["feature_cols"]
    x = np.array([[features.get(col, 0.0) for col in cols]], dtype=np.float32)
    x = model_bundle["scaler"].transform(x)
    probs = model.predict_proba(x)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "label": str(encoder.inverse_transform([idx])[0]),
            "confidence": float(round(probs[idx], 5)),
        }
        for idx in order
    ]


def write_html(result: dict, path: Path) -> None:
    rows = []
    for segment in result["segments"]:
        label_text = "<br>".join(
            f"{html.escape(item['label'])} ({item['confidence']:.3f})" for item in segment["predictions"].get("label", [])
        )
        group_text = "<br>".join(
            f"{html.escape(item['label'])} ({item['confidence']:.3f})" for item in segment["predictions"].get("group", [])
        )
        rows.append(
            "<tr>"
            f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
            f"<td>{group_text}</td>"
            f"<td>{label_text}</td>"
            f"<td>{segment['stats']['centroid']:.0f}</td>"
            f"<td>{segment['stats']['flatness']:.3f}</td>"
            f"<td>{segment['stats']['motion_strength']:.3f}</td>"
            f"<td>{segment['stats']['width']:.3f}</td>"
            "</tr>"
        )
    body = "\n".join(rows)
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Reference Element Timeline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 8px; vertical-align: top; }}
th {{ background: #f0f0f0; text-align: left; }}
.note {{ margin: 16px 0; color: #444; }}
</style>
<h1>Reference Element Timeline</h1>
<p><b>Audio:</b> {html.escape(result['audio'])}</p>
<p class="note">This is a synthetic-reference classifier panel. Treat labels as attraction toward a learned sound family, not exact source separation.</p>
<table>
<thead><tr><th>Time</th><th>Group</th><th>Element Label</th><th>Brightness</th><th>Noise</th><th>Motion</th><th>Width</th></tr></thead>
<tbody>{body}</tbody>
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/reference_element_fast_synth_v1.joblib"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reference_element_timeline.json"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/reference_element_timeline.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    audio, sr = load_audio(args.audio)
    segment_len = max(1, int(args.segment_seconds * sr))
    hop_len = max(1, int(args.hop_seconds * sr))
    mono = audio.mean(axis=1)
    duration = len(mono) / sr
    starts = list(range(0, max(1, len(mono) - segment_len + 1), hop_len))
    if not starts:
        starts = [0]

    segments = []
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, start in enumerate(starts):
            end = min(len(audio), start + segment_len)
            clip = audio[start:end]
            if len(clip) < segment_len:
                pad = np.zeros((segment_len - len(clip), clip.shape[1]), dtype=np.float32)
                clip = np.vstack([clip, pad])
            clip_path = tmp_dir / f"segment_{idx:04d}.wav"
            sf.write(clip_path, clip, sr)
            stats = audio_stats(clip_path, quality=args.quality)
            predictions = {}
            for target in bundle.get("targets", []):
                predictions[target] = top_probs(bundle, stats, target, args.top_k)
            segments.append(
                {
                    "index": idx,
                    "start": round(start / sr, 4),
                    "end": round(min(duration, (start + segment_len) / sr), 4),
                    "predictions": predictions,
                    "stats": stats,
                }
            )

    result = {
        "audio": str(args.audio),
        "model": str(args.model),
        "segment_seconds": args.segment_seconds,
        "hop_seconds": args.hop_seconds,
        "segments": segments,
        "caution": "Synthetic-reference labels are not exact source separation or plugin recovery.",
    }
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(result, args.out_html)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
