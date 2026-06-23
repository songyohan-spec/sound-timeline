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
from infer_reference_elements_timeline import load_audio


def multilabel_probs(bundle: dict, features: dict[str, float], output_name: str, top_k: int, threshold: float) -> list[dict]:
    entry = bundle["outputs"][output_name]
    model = entry["model"]
    binarizer = entry["binarizer"]
    cols = bundle["feature_cols"]
    x = np.array([[features.get(col, 0.0) for col in cols]], dtype=np.float32)
    x = bundle["scaler"].transform(x)
    probs_raw = model.predict_proba(x)
    probs = []
    for idx, raw in enumerate(probs_raw):
        classes = model.classes_[idx]
        if len(classes) == 1:
            prob = float(classes[0])
        else:
            one_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            prob = float(raw[0][one_index])
        probs.append(prob)
    order = np.argsort(probs)[::-1]
    rows = []
    for idx in order[:top_k]:
        label = str(binarizer.classes_[idx])
        calibrated = bundle.get("thresholds", {}).get(output_name, {}).get(label, threshold)
        calibrated = max(float(calibrated), float(threshold))
        status = "detected" if probs[idx] >= calibrated else "possible"
        rows.append(
            {
                "label": label,
                "confidence": round(probs[idx], 5),
                "threshold": round(float(calibrated), 5),
                "status": status,
            }
        )
    return rows


def write_html(result: dict, path: Path, threshold: float) -> None:
    rows = []
    for segment in result["segments"]:
        groups = "<br>".join(
            f"{html.escape(item['label'])} ({item['confidence']:.3f}, {item['status']})"
            for item in segment["groups"]
        )
        labels = "<br>".join(
            f"{html.escape(item['label'])} ({item['confidence']:.3f}, {item['status']})"
            for item in segment["labels"]
        )
        detected = ", ".join(item["label"] for item in segment["labels"] if item["confidence"] >= threshold) or "-"
        rows.append(
            "<tr>"
            f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
            f"<td>{html.escape(detected)}</td>"
            f"<td>{groups}</td>"
            f"<td>{labels}</td>"
            f"<td>{segment['stats']['centroid']:.0f}</td>"
            f"<td>{segment['stats']['flatness']:.3f}</td>"
            f"<td>{segment['stats']['motion_strength']:.3f}</td>"
            f"<td>{segment['stats']['width']:.3f}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Multi-Label Sound Element Timeline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 8px; vertical-align: top; }}
th {{ background: #f0f0f0; text-align: left; }}
.note {{ margin: 16px 0; color: #444; }}
</style>
<h1>Multi-Label Sound Element Timeline</h1>
<p><b>Audio:</b> {html.escape(result['audio'])}</p>
<p class="note">Threshold: {threshold}. Labels below threshold are still shown as possible context, not hard detections.</p>
<table>
<thead><tr><th>Time</th><th>Detected Elements</th><th>Group Candidates</th><th>Element Candidates</th><th>Brightness</th><th>Noise</th><th>Motion</th><th>Width</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
<p>This model was trained on synthetic mixtures. It is a structural prototype, not a finished recognizer.</p>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/reference_mixture_multilabel_v1.joblib"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/reference_mixture_timeline.json"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/reference_mixture_timeline.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    audio, sr = load_audio(args.audio)
    segment_len = max(1, int(args.segment_seconds * sr))
    hop_len = max(1, int(args.hop_seconds * sr))
    duration = len(audio) / sr
    starts = list(range(0, max(1, len(audio) - segment_len + 1), hop_len)) or [0]
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
            segments.append(
                {
                    "index": idx,
                    "start": round(start / sr, 4),
                    "end": round(min(duration, (start + segment_len) / sr), 4),
                    "groups": multilabel_probs(bundle, stats, "groups", args.top_k, args.threshold),
                    "labels": multilabel_probs(bundle, stats, "labels", args.top_k, args.threshold),
                    "stats": stats,
                }
            )
    result = {
        "audio": str(args.audio),
        "model": str(args.model),
        "threshold": args.threshold,
        "segments": segments,
    }
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(result, args.out_html, args.threshold)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
