from __future__ import annotations

import argparse
import csv
import html
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats


def positive_probabilities(bundle: dict, features: dict[str, float], output_name: str) -> dict[str, float]:
    entry = bundle["outputs"][output_name]
    model = entry["model"]
    binarizer = entry["binarizer"]
    cols = bundle["feature_cols"]
    x = np.array([[features.get(col, 0.0) for col in cols]], dtype=np.float32)
    x = bundle["scaler"].transform(x)
    raw = model.predict_proba(x)
    probs = {}
    for idx, class_probs in enumerate(raw):
        classes = model.classes_[idx]
        if len(classes) == 1:
            prob = float(classes[0])
        else:
            one_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            prob = float(class_probs[0][one_index])
        probs[str(binarizer.classes_[idx])] = prob
    return probs


def load_heuristic(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            out[(row["file"], str(row["start"]), str(row["end"]))] = row
    return out


def format_scores(items: list[tuple[str, float]], thresholds: dict[str, float], floor: float, limit: int) -> str:
    parts = []
    for label, score in items[:limit]:
        threshold = max(float(thresholds.get(label, floor)), floor)
        status = "detected" if score >= threshold else "possible"
        parts.append(f"{label}:{score:.3f}/{threshold:.3f}/{status}")
    return "; ".join(parts)


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def write_html(rows: list[dict], out_html: Path) -> None:
    agreement_counter = Counter(row["agreement"] for row in rows)
    detected_counter = Counter()
    for row in rows:
        detected_counter.update(split_pipe(row["model_detected"]))

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    details = []
    for row in rows:
        details.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
            f"<td>{html.escape(row['heuristic_active'] or '-')}</td>"
            f"<td>{html.escape(row['model_detected'] or '-')}</td>"
            f"<td>{html.escape(row['overlap'] or '-')}</td>"
            f"<td>{html.escape(row['agreement'])}</td>"
            f"<td>{html.escape(row['model_top'])}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Source Kind Model Check</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
section {{ border-top: 2px solid #111; margin-top: 26px; padding-top: 10px; }}
</style>
<h1>Source Kind Model Check</h1>
<p>This compares the heuristic/external-panel source-kind report against a trained source-kind multi-label model. Agreement is not ground truth, but disagreement tells us where the label/data/model loop is weak.</p>
{count_table("Agreement", agreement_counter)}
{count_table("Model Detected Labels", detected_counter)}
<h2>Segment Detail</h2>
<table>
<tr><th>Segment</th><th>Audio</th><th>Heuristic Active</th><th>Model Detected</th><th>Overlap</th><th>Agreement</th><th>Model Top Scores</th></tr>
{''.join(details)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/review_queue.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/source_kind_multilabel_v1.joblib"))
    parser.add_argument("--heuristic", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.csv"))
    parser.add_argument("--queue-root", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue"))
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--floor", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_model_check.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_model_check.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    heuristic = load_heuristic(args.heuristic)
    thresholds = bundle.get("thresholds", {}).get("labels", {})

    with args.queue.open("r", encoding="utf-8-sig", newline="") as file:
        queue_rows = list(csv.DictReader(file))

    out_rows = []
    for index, row in enumerate(queue_rows, 1):
        clip = row.get("clip", "")
        clip_path = Path(clip)
        if not clip_path.is_absolute():
            clip_path = args.queue_root / clip_path
        stats = audio_stats(clip_path, quality=args.quality)
        probs = positive_probabilities(bundle, stats, "labels")
        ordered = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        detected = []
        for label, score in ordered:
            threshold = max(float(thresholds.get(label, args.floor)), args.floor)
            if score >= threshold:
                detected.append(label)

        key = (row["file"], str(row["start"]), str(row["end"]))
        heur = heuristic.get(key, {})
        heuristic_active = split_pipe(heur.get("active_source_kinds", ""))
        overlap = sorted(set(heuristic_active) & set(detected))
        if overlap:
            agreement = "overlap"
        elif detected and heuristic_active:
            agreement = "disagree"
        elif detected:
            agreement = "model_only"
        elif heuristic_active:
            agreement = "heuristic_only"
        else:
            agreement = "empty"

        out_rows.append(
            {
                "file": row["file"],
                "start": row["start"],
                "end": row["end"],
                "clip": clip,
                "heuristic_active": "|".join(heuristic_active),
                "model_detected": "|".join(detected),
                "overlap": "|".join(overlap),
                "agreement": agreement,
                "model_top": format_scores(ordered, thresholds, args.floor, args.top_k),
            }
        )
        if index % 50 == 0:
            print(f"processed {index}/{len(queue_rows)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    write_html(out_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
