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
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats
from infer_reference_elements_timeline import load_audio, top_probs


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def analyze_file(bundle: dict, path: Path, segment_seconds: float, hop_seconds: float, top_k: int, quality: str) -> dict:
    import numpy as np

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
            predictions = {
                target: top_probs(bundle, stats, target, top_k)
                for target in bundle.get("targets", [])
            }
            segments.append(
                {
                    "index": idx,
                    "start": round(start / sr, 4),
                    "end": round(min(duration, (start + segment_len) / sr), 4),
                    "predictions": predictions,
                    "stats": stats,
                }
            )
    return {"audio": str(path), "segments": segments}


def write_summary(results: list[dict], csv_path: Path, html_path: Path) -> None:
    rows = []
    group_counter: Counter[str] = Counter()
    label_counter: Counter[str] = Counter()
    for result in results:
        file_name = Path(result["audio"]).name
        for segment in result["segments"]:
            top_group = segment["predictions"].get("group", [{}])[0]
            top_label = segment["predictions"].get("label", [{}])[0]
            group = top_group.get("label", "")
            label = top_label.get("label", "")
            group_counter[group] += 1
            label_counter[label] += 1
            rows.append(
                {
                    "file": file_name,
                    "start": segment["start"],
                    "end": segment["end"],
                    "group": group,
                    "group_confidence": top_group.get("confidence", ""),
                    "label": label,
                    "label_confidence": top_label.get("confidence", ""),
                    "top3_labels": "; ".join(
                        f"{item['label']}:{item['confidence']:.3f}"
                        for item in segment["predictions"].get("label", [])[:3]
                    ),
                }
            )

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "start", "end", "group", "group_confidence", "label", "label_confidence", "top3_labels"],
        )
        writer.writeheader()
        writer.writerows(rows)

    def counter_table(title: str, counter: Counter[str]) -> str:
        body = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
            for label, count in counter.most_common()
        )
        return f"<h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table>"

    detail_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['file']))}</td>"
        f"<td>{row['start']:.2f}-{row['end']:.2f}s</td>"
        f"<td>{html.escape(str(row['group']))} ({row['group_confidence']})</td>"
        f"<td>{html.escape(str(row['label']))} ({row['label_confidence']})</td>"
        f"<td>{html.escape(str(row['top3_labels']))}</td>"
        "</tr>"
        for row in rows
    )
    html_text = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Batch Reference Element Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; margin: 12px 0 28px; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; text-align: left; vertical-align: top; }}
th {{ background: #f0f0f0; }}
</style>
<h1>Batch Reference Element Summary</h1>
<p>Segments: {len(rows)}</p>
{counter_table("Top Groups", group_counter)}
{counter_table("Top Element Labels", label_counter)}
<h2>Details</h2>
<table><tr><th>File</th><th>Time</th><th>Group</th><th>Top Label</th><th>Top 3 Labels</th></tr>{detail_rows}</table>
<p>These labels come from a synthetic-reference model and should be used as attraction cues, not exact source separation.</p>
</html>"""
    html_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/reference_element_fast_synth_v1.joblib"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/external_reference_elements_batch.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/external_reference_elements_batch.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/external_reference_elements_batch.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    files = discover_audio(args.input_dir)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")
    results = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path.name}")
        results.append(analyze_file(bundle, path, args.segment_seconds, args.hop_seconds, args.top_k, args.quality))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps({"files": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(results, args.out_csv, args.out_html)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
