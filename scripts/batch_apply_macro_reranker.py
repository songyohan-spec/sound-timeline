from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def counter_table(title: str, counter: Counter[str]) -> str:
    body = "\n".join(f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in counter.most_common())
    return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-dir", type=Path, default=Path("outputs/palette_timelines_v3"))
    parser.add_argument("--model", type=Path, default=Path("models/macro_reranker_v3.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/reranked_timelines_v3"))
    args = parser.parse_args()

    paths = sorted(args.timeline_dir.glob("*_palette_timeline.json"))
    if not paths:
        raise SystemExit(f"No palette timeline JSON files found in {args.timeline_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for path in paths:
        out_json = args.out_dir / path.name.replace("_palette_timeline.json", "_reranked_timeline.json")
        out_html = args.out_dir / path.name.replace("_palette_timeline.json", "_reranked_timeline.html")
        run(
            [
                sys.executable,
                "scripts/apply_macro_reranker.py",
                "--timeline",
                str(path),
                "--model",
                str(args.model),
                "--out-json",
                str(out_json),
                "--out-html",
                str(out_html),
            ]
        )
        output_paths.append(out_json)

    source_counts: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    region_source_counts: Counter[str] = Counter()
    region_processing_counts: Counter[str] = Counter()
    rows = []
    for path in output_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        for segment in report.get("segments", []):
            source_counts[segment["source_macro"]["label"]] += 1
            processing_counts[segment["processing_macro"]["label"]] += 1
        for region in report.get("regions", []):
            source = region["source_macro"]
            processing = region["processing_macro"]
            region_source_counts[source["label"]] += 1
            region_processing_counts[processing["label"]] += 1
            rows.append(
                {
                    "file": path.name,
                    "time": f"{region['start']:.2f}-{region['end']:.2f}s",
                    "source": f"{source['label']} ({source['confidence']:.2f})",
                    "processing": f"{processing['label']} ({processing['confidence']:.2f})",
                }
            )

    summary = {
        "timeline_dir": str(args.timeline_dir),
        "model": str(args.model),
        "source_counts": dict(source_counts.most_common()),
        "processing_counts": dict(processing_counts.most_common()),
        "region_source_counts": dict(region_source_counts.most_common()),
        "region_processing_counts": dict(region_processing_counts.most_common()),
        "rows": rows,
    }
    summary_json = args.out_dir / "reranked_timeline_batch_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_rows = "\n".join(
        "<tr>"
        f"<td>{row['file']}</td><td>{row['time']}</td><td>{row['source']}</td><td>{row['processing']}</td>"
        "</tr>"
        for row in rows
    )
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Reranked Timeline Batch Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>Reranked Timeline Batch Summary</h1>",
            "<div class='grid'>",
            counter_table("Segment Source Macro", source_counts),
            counter_table("Segment Processing Macro", processing_counts),
            counter_table("Region Source Macro", region_source_counts),
            counter_table("Region Processing Macro", region_processing_counts),
            "</div>",
            "<h2>Regions</h2>",
            "<table><thead><tr><th>Timeline</th><th>Time</th><th>Source Macro</th><th>Processing Macro</th></tr></thead><tbody>",
            detail_rows,
            "</tbody></table>",
        ]
    )
    summary_html = args.out_dir / "reranked_timeline_batch_summary.html"
    summary_html.write_text(html, encoding="utf-8")
    print(f"summary: {summary_json}")
    print(f"summary: {summary_html}")


if __name__ == "__main__":
    main()
