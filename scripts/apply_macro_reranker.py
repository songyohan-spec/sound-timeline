from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np


def score_features(profile: dict) -> dict[str, float]:
    features: dict[str, float] = {}
    for item in profile.get("ranked_palette", []):
        family = item.get("family", "unknown")
        label = item.get("label", "unknown")
        features[f"score::{family}::{label}"] = float(item.get("score") or 0.0)
    return features


def predict_one(bundle: dict, profile: dict, target: str) -> dict:
    features = score_features(profile)
    x = np.array([[features.get(name, 0.0) for name in bundle["features"]]], dtype=float)
    model = bundle["models"][target]
    encoder = bundle["encoders"][target]
    probs = model.predict_proba(x)[0]
    order = probs.argsort()[::-1]
    raw_confidence = float(probs[order[0]])
    cap = float(bundle.get("confidence_caps", {}).get(target, 1.0))
    return {
        "label": str(encoder.inverse_transform([order[0]])[0]),
        "confidence": round(min(raw_confidence, cap), 4),
        "raw_confidence": round(raw_confidence, 4),
        "confidence_cap": round(cap, 4),
        "alternatives": [
            {
                "label": str(encoder.inverse_transform([index])[0]),
                "confidence": round(float(probs[index]), 4),
            }
            for index in order[1:3]
        ],
    }


def build_regions(segments: list[dict]) -> list[dict]:
    regions = []
    for segment in segments:
        key = (segment["source_macro"]["label"], segment["processing_macro"]["label"])
        if regions and regions[-1]["key"] == key and abs(regions[-1]["end"] - segment["start"]) < 0.001:
            regions[-1]["end"] = segment["end"]
            regions[-1]["segments"].append(segment["index"])
            continue
        regions.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "key": key,
                "source_macro": segment["source_macro"],
                "processing_macro": segment["processing_macro"],
                "segments": [segment["index"]],
            }
        )
    for region in regions:
        region.pop("key", None)
    return regions


def render_html(report: dict) -> str:
    region_rows = []
    for region in report["regions"]:
        region_rows.append(
            "<tr>"
            f"<td>{region['start']:.2f}-{region['end']:.2f}s</td>"
            f"<td>{region['source_macro']['label']} ({region['source_macro']['confidence']:.2f})</td>"
            f"<td>{region['processing_macro']['label']} ({region['processing_macro']['confidence']:.2f})</td>"
            f"<td>{', '.join(str(index) for index in region['segments'])}</td>"
            "</tr>"
        )
    segment_rows = []
    for segment in report["segments"]:
        segment_rows.append(
            "<tr>"
            f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
            f"<td>{segment['source_macro']['label']} ({segment['source_macro']['confidence']:.2f})</td>"
            f"<td>{segment['processing_macro']['label']} ({segment['processing_macro']['confidence']:.2f})</td>"
            f"<td>{segment['raw_primary_source']}</td>"
            f"<td>{segment['raw_primary_processing']}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            f"<title>{report['title']} Reranked Macro Timeline</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.note{color:#555}</style>",
            f"<h1>{report['title']} Reranked Macro Timeline</h1>",
            "<p class='note'>Tiny reranker trained on pseudo-calibrated CLAP outputs. Treat as a prototype, not ground truth.</p>",
            "<h2>Region Summary</h2>",
            "<table><thead><tr><th>Time</th><th>Source Macro</th><th>Processing Macro</th><th>Segments</th></tr></thead><tbody>",
            *region_rows,
            "</tbody></table>",
            "<h2>Segment Detail</h2>",
            "<table><thead><tr><th>Time</th><th>Source Macro</th><th>Processing Macro</th><th>Raw Primary Source</th><th>Raw Primary Processing</th></tr></thead><tbody>",
            *segment_rows,
            "</tbody></table>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/macro_reranker_v3.joblib"))
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    timeline = json.loads(args.timeline.read_text(encoding="utf-8"))
    segments = []
    for segment in timeline.get("segments", []):
        profile = segment.get("profile", {})
        raw_primary_source = profile.get("primary_sound") or {}
        raw_primary_processing = profile.get("primary_processing") or {}
        segments.append(
            {
                "index": segment.get("index"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "source_macro": predict_one(bundle, profile, "target_source_macro"),
                "processing_macro": predict_one(bundle, profile, "target_processing_macro"),
                "raw_primary_source": raw_primary_source.get("label", "none"),
                "raw_primary_processing": raw_primary_processing.get("label", "none"),
            }
        )

    report = {
        "title": timeline.get("title", args.timeline.stem),
        "audio": timeline.get("audio"),
        "mode": "learned_macro_reranker_timeline",
        "model": str(args.model),
        "regions": build_regions(segments),
        "segments": segments,
    }
    out_json = args.out_json or args.timeline.with_name(args.timeline.stem.replace("_palette_timeline", "_reranked_timeline") + ".json")
    out_html = args.out_html or args.timeline.with_name(args.timeline.stem.replace("_palette_timeline", "_reranked_timeline") + ".html")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_html.write_text(render_html(report), encoding="utf-8")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()
