from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from infer_reference_mixture_timeline import main as _unused_main
from infer_reference_mixture_timeline import load_audio, multilabel_probs
from dsp_palette_score import audio_stats

import joblib
import numpy as np
import soundfile as sf
import tempfile


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def analyze_file(bundle: dict, path: Path, segment_seconds: float, hop_seconds: float, threshold: float, top_k: int, quality: str) -> dict:
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
                    "groups": multilabel_probs(bundle, stats, "groups", top_k, threshold),
                    "labels": multilabel_probs(bundle, stats, "labels", top_k, threshold),
                    "stats": stats,
                }
            )
    return {"audio": str(path), "segments": segments}


def detected_labels(items: list[dict], threshold: float) -> list[str]:
    labels = [item["label"] for item in items if item.get("status") == "detected"]
    if labels:
        return labels
    return [item["label"] for item in items if float(item["confidence"]) >= threshold]


def write_outputs(results: list[dict], out_csv: Path, out_html: Path, threshold: float) -> None:
    rows = []
    group_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    possible_counter: Counter[str] = Counter()

    for result in results:
        file_name = Path(result["audio"]).name
        for seg in result["segments"]:
            detected_groups = detected_labels(seg["groups"], threshold)
            detected = detected_labels(seg["labels"], threshold)
            for group in detected_groups:
                group_counter[group] += 1
            for label in detected:
                label_counter[label] += 1
            for item in seg["labels"][:5]:
                possible_counter[item["label"]] += 1
            rows.append(
                {
                    "file": file_name,
                    "start": seg["start"],
                    "end": seg["end"],
                    "detected_groups": "|".join(detected_groups),
                    "detected_labels": "|".join(detected),
                    "top_groups": "; ".join(f"{x['label']}:{x['confidence']:.3f}" for x in seg["groups"][:5]),
                    "top_labels": "; ".join(f"{x['label']}:{x['confidence']:.3f}" for x in seg["labels"][:8]),
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
        f"<td>{html.escape(str(row['detected_groups']) or '-')}</td>"
        f"<td>{html.escape(str(row['detected_labels']) or '-')}</td>"
        f"<td>{html.escape(str(row['top_labels']))}</td>"
        "</tr>"
        for row in rows
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Batch Multi-Label Sound Elements</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; }}
</style>
<h1>Batch Multi-Label Sound Elements</h1>
<p>Segments: {len(rows)} | Fallback threshold: {threshold}</p>
<p class="note">Uses calibrated per-label thresholds when present. Synthetic mixture model; read as sound-element attraction cues, not exact source separation.</p>
{table("Detected Groups", group_counter)}
{table("Detected Element Labels", label_counter)}
{table("Top-5 Label Mentions", possible_counter)}
<h2>Details</h2>
<table>
<tr><th>File</th><th>Time</th><th>Detected Groups</th><th>Detected Labels</th><th>Top Label Candidates</th></tr>
{detail}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/reference_mixture_multilabel_v1.joblib"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/external_reference_mixture_batch.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/external_reference_mixture_batch.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/external_reference_mixture_batch.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    files = discover_audio(args.input_dir)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")
    results = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        results.append(analyze_file(bundle, path, args.segment_seconds, args.hop_seconds, args.threshold, args.top_k, args.quality))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"files": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_outputs(results, args.out_csv, args.out_html, args.threshold)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
