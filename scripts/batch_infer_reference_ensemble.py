from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats
from generate_reference_synthetic_dataset import GROUP_BY_LABEL
from infer_reference_mixture_timeline import load_audio


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}
VOCAL_SENSITIVE_LABELS = {
    "processed_lead_vocal",
    "hard_tuned_vocal",
    "pitched_vocal_chop",
    "breathy_vocal_pad",
    "stacked_harmony_vocal",
    "vocal_synth_hybrid",
}


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def positive_probabilities(bundle: dict, stats: dict[str, float], output_name: str) -> dict[str, float]:
    entry = bundle["outputs"][output_name]
    model = entry["model"]
    binarizer = entry["binarizer"]
    cols = bundle["feature_cols"]
    x = np.array([[stats.get(col, 0.0) for col in cols]], dtype=np.float32)
    x = bundle["scaler"].transform(x)
    raw = model.predict_proba(x)
    probs: dict[str, float] = {}
    for idx, class_probs in enumerate(raw):
        classes = model.classes_[idx]
        if len(classes) == 1:
            prob = float(classes[0])
        else:
            one_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            prob = float(class_probs[0][one_index])
        probs[str(binarizer.classes_[idx])] = prob
    return probs


def threshold_for(bundles: list[dict], output_name: str, label: str, floor: float) -> float:
    values = []
    for bundle in bundles:
        value = bundle.get("thresholds", {}).get(output_name, {}).get(label)
        if value is not None:
            values.append(float(value))
    if not values:
        return floor
    return max(floor, float(np.mean(values)))


def ensemble_scores(
    bundles: list[dict],
    stats: dict[str, float],
    output_name: str,
    v4_weight: float,
    vocal_v5_boost: float,
    top_k: int,
    floor: float,
) -> list[dict]:
    per_model = [positive_probabilities(bundle, stats, output_name) for bundle in bundles]
    labels = sorted(set().union(*(scores.keys() for scores in per_model)))
    rows = []
    for label in labels:
        if len(per_model) == 1:
            score = per_model[0].get(label, 0.0)
        else:
            w4 = v4_weight
            w5 = 1.0 - v4_weight
            if output_name == "labels" and label in VOCAL_SENSITIVE_LABELS:
                w5 *= vocal_v5_boost
            total = w4 + w5
            score = (w4 * per_model[0].get(label, 0.0) + w5 * per_model[1].get(label, 0.0)) / total
        threshold = threshold_for(bundles, output_name, label, floor)
        rows.append(
            {
                "label": label,
                "confidence": round(float(score), 5),
                "threshold": round(float(threshold), 5),
                "status": "detected" if score >= threshold else "possible",
            }
        )
    rows.sort(key=lambda item: item["confidence"], reverse=True)
    return rows[:top_k]


def analyze_file(
    bundles: list[dict],
    path: Path,
    segment_seconds: float,
    hop_seconds: float,
    floor: float,
    top_k: int,
    quality: str,
    v4_weight: float,
    vocal_v5_boost: float,
) -> dict:
    audio, sr = load_audio(path)
    segment_len = max(1, int(segment_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    duration = len(audio) / sr
    starts = list(range(0, max(1, len(audio) - segment_len + 1), hop_len)) or [0]
    segments = []
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
            stats = audio_stats(clip_path, quality=quality)
            segments.append(
                {
                    "index": idx,
                    "start": round(start / sr, 4),
                    "end": round(min(duration, (start + segment_len) / sr), 4),
                    "groups": ensemble_scores(bundles, stats, "groups", v4_weight, vocal_v5_boost, top_k, floor),
                    "labels": ensemble_scores(bundles, stats, "labels", v4_weight, vocal_v5_boost, top_k, floor),
                    "stats": stats,
                }
            )
    return {"audio": str(path), "segments": segments}


def detected(items: list[dict]) -> list[str]:
    return [item["label"] for item in items if item.get("status") == "detected"]


def write_outputs(results: list[dict], out_csv: Path, out_html: Path) -> None:
    rows = []
    group_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    top_counter: Counter[str] = Counter()
    for result in results:
        file_name = Path(result["audio"]).name
        for seg in result["segments"]:
            labels = detected(seg["labels"])
            # Only promote a broad group to "detected" when at least one
            # concrete element label in that group is detected. The independent
            # group head is kept in top_groups as a weak broad cue.
            groups = sorted({GROUP_BY_LABEL.get(label, "unknown") for label in labels})
            for group in groups:
                group_counter[group] += 1
            for label in labels:
                label_counter[label] += 1
            for item in seg["labels"][:5]:
                top_counter[item["label"]] += 1
            rows.append(
                {
                    "file": file_name,
                    "start": seg["start"],
                    "end": seg["end"],
                    "detected_groups": "|".join(groups),
                    "detected_labels": "|".join(labels),
                    "top_groups": "; ".join(f"{x['label']}:{x['confidence']:.3f}/{x['threshold']:.3f}" for x in seg["groups"][:5]),
                    "top_labels": "; ".join(f"{x['label']}:{x['confidence']:.3f}/{x['threshold']:.3f}" for x in seg["labels"][:8]),
                    "brightness_centroid": round(seg["stats"]["centroid"], 4),
                    "flatness_noise": round(seg["stats"]["flatness"], 6),
                    "motion_strength": round(seg["stats"]["motion_strength"], 6),
                    "width": round(seg["stats"]["width"], 6),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = [
            "file",
            "start",
            "end",
            "detected_groups",
            "detected_labels",
            "top_groups",
            "top_labels",
            "brightness_centroid",
            "flatness_noise",
            "motion_strength",
            "width",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table>"

    detail = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['file']))}</td>"
        f"<td>{row['start']:.2f}-{row['end']:.2f}s</td>"
        f"<td>{html.escape(row['detected_groups'] or '-')}</td>"
        f"<td>{html.escape(row['detected_labels'] or '-')}</td>"
        f"<td>{html.escape(row['top_labels'])}</td>"
        "</tr>"
        for row in rows
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Reference Ensemble Sound Elements</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; }}
</style>
<h1>Reference Ensemble Sound Elements</h1>
<p>Segments: {len(rows)}</p>
<p class="note">Ensemble of v4 bands and v5 vocal-focused models. Scores are still synthetic-reference cues, not source separation.</p>
{table("Detected Groups", group_counter)}
{table("Detected Element Labels", label_counter)}
{table("Top-5 Label Mentions", top_counter)}
<h2>Details</h2>
<table>
<tr><th>File</th><th>Time</th><th>Detected Groups</th><th>Detected Labels</th><th>Top Label Candidates score/threshold</th></tr>
{detail}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", type=Path, default=[
        Path("models/reference_mixture_multilabel_v4_bands.joblib"),
        Path("models/reference_mixture_multilabel_v5_vocal.joblib"),
    ])
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold-floor", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--v4-weight", type=float, default=0.55)
    parser.add_argument("--vocal-v5-boost", type=float, default=1.35)
    parser.add_argument("--out-json", type=Path, default=Path("outputs/external_reference_ensemble_batch.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/external_reference_ensemble_batch.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/external_reference_ensemble_batch.html"))
    args = parser.parse_args()

    bundles = [joblib.load(path) for path in args.models]
    files = discover_audio(args.input_dir)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")
    results = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        results.append(
            analyze_file(
                bundles,
                path,
                args.segment_seconds,
                args.hop_seconds,
                args.threshold_floor,
                args.top_k,
                args.quality,
                args.v4_weight,
                args.vocal_v5_boost,
            )
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"files": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_outputs(results, args.out_csv, args.out_html)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
