import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from dsp_palette_score import audio_stats


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def load_audio(path: Path):
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), sr


def predict(stats, model, top_k=3):
    x = np.array([float(stats[col]) for col in model["feature_columns"]], dtype=np.float64)
    z = (x - np.array(model["mean"], dtype=np.float64)) / np.array(model["std"], dtype=np.float64)
    scored = []
    for cid, item in model["clusters"].items():
        centroid = np.array(item["centroid"], dtype=np.float64)
        distance = float(np.sqrt(np.sum((z - centroid) ** 2)))
        confidence = float(1.0 / (1.0 + distance))
        scored.append(
            {
                "cluster": int(cid),
                "label": item["label"],
                "distance": round(distance, 4),
                "confidence": round(confidence, 4),
                "notes": item.get("notes", ""),
            }
        )
    return sorted(scored, key=lambda row: row["distance"])[:top_k]


def merge_regions(segments):
    regions = []
    for seg in segments:
        top = seg["predictions"][0]
        key = top["label"]
        if regions and regions[-1]["label"] == key and abs(regions[-1]["end"] - seg["start"]) < 1e-6:
            regions[-1]["end"] = seg["end"]
            regions[-1]["segments"].append(seg["index"])
            regions[-1]["mean_confidence"] = round(
                (regions[-1]["mean_confidence"] * (len(regions[-1]["segments"]) - 1) + top["confidence"])
                / len(regions[-1]["segments"]),
                4,
            )
        else:
            regions.append(
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "label": key,
                    "cluster": top["cluster"],
                    "mean_confidence": top["confidence"],
                    "segments": [seg["index"]],
                }
            )
    return regions


def analyze(audio_path, model, segment_seconds, hop_seconds, quality):
    audio, sr = load_audio(audio_path)
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(audio) < seg_len:
        audio = np.pad(audio, ((0, seg_len - len(audio)), (0, 0)))

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        index = 0
        start = 0
        while start + seg_len <= len(audio):
            end = start + seg_len
            segment_path = tmp_dir / f"segment_{index:04d}.wav"
            sf.write(segment_path, audio[start:end], sr)
            stats = audio_stats(segment_path, quality=quality)
            predictions = predict(stats, model)
            segments.append(
                {
                    "index": index,
                    "start": round(start / sr, 3),
                    "end": round(end / sr, 3),
                    "predictions": predictions,
                    "stats": stats,
                }
            )
            index += 1
            start += hop_len
    return {
        "audio": str(audio_path),
        "mode": "dsp_palette_centroid_timeline",
        "segment_seconds": segment_seconds,
        "hop_seconds": hop_seconds,
        "regions": merge_regions(segments),
        "segments": segments,
        "caution": model.get("caution", ""),
    }


def write_csv(report, path):
    fieldnames = ["audio", "start", "end", "label", "cluster", "confidence", "alternatives"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in report["segments"]:
            top = seg["predictions"][0]
            alts = "; ".join(f"{p['label']}:{p['confidence']}" for p in seg["predictions"][1:])
            writer.writerow(
                {
                    "audio": report["audio"],
                    "start": f"{seg['start']:.2f}",
                    "end": f"{seg['end']:.2f}",
                    "label": top["label"],
                    "cluster": top["cluster"],
                    "confidence": top["confidence"],
                    "alternatives": alts,
                }
            )


def write_html(report, path):
    rows = []
    for region in report["regions"]:
        rows.append(
            "<tr>"
            f"<td>{region['start']:.2f}-{region['end']:.2f}s</td>"
            f"<td>{region['label']}</td>"
            f"<td>{region['cluster']}</td>"
            f"<td>{region['mean_confidence']}</td>"
            f"<td>{', '.join(str(s) for s in region['segments'])}</td>"
            "</tr>"
        )
    detail_rows = []
    for seg in report["segments"]:
        pred = "<br>".join(f"{p['label']} ({p['confidence']})" for p in seg["predictions"])
        detail_rows.append(
            "<tr>"
            f"<td>{seg['start']:.2f}-{seg['end']:.2f}s</td>"
            f"<td>{pred}</td>"
            "</tr>"
        )
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#111}table{border-collapse:collapse;width:100%;margin-bottom:28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#eee}.note{color:#555}</style>",
            "<h1>Sound Palette Centroid Timeline</h1>",
            f"<p class='note'>Audio: {report['audio']}</p>",
            "<h2>Region Summary</h2>",
            "<table><thead><tr><th>Time</th><th>Sound Palette</th><th>Cluster</th><th>Mean Confidence</th><th>Segments</th></tr></thead><tbody>",
            *rows,
            "</tbody></table>",
            "<h2>Segment Alternatives</h2>",
            "<table><thead><tr><th>Time</th><th>Top Matches</th></tr></thead><tbody>",
            *detail_rows,
            "</tbody></table>",
            f"<p class='note'>{report.get('caution', '')}</p>",
        ]
    )
    Path(path).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--quality", choices=["librosa", "fast"], default="librosa")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--out-html", default=None)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    report = analyze(audio_path, model, args.segment_seconds, args.hop_seconds, args.quality)
    stem = safe_stem(audio_path)
    out_json = Path(args.out_json or f"outputs/{stem}_centroid_timeline.json")
    out_csv = Path(args.out_csv or f"outputs/{stem}_centroid_timeline.csv")
    out_html = Path(args.out_html or f"outputs/{stem}_centroid_timeline.html")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(report, out_csv)
    write_html(report, out_html)
    print(f"wrote: {out_json}")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()
