from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calibrate_palette_timeline import calibrate, render_html


def counter_table(title: str, counter: Counter[str]) -> str:
    body = "\n".join(f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in counter.most_common())
    return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-dir", type=Path, default=Path("outputs/palette_timelines_v3"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/calibrated_timelines"))
    args = parser.parse_args()

    paths = sorted(args.timeline_dir.glob("*_palette_timeline.json"))
    if not paths:
        raise SystemExit(f"No palette timeline JSON files found in {args.timeline_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source_counts: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    region_source_counts: Counter[str] = Counter()
    region_processing_counts: Counter[str] = Counter()
    rows = []

    for path in paths:
        timeline = json.loads(path.read_text(encoding="utf-8"))
        report = calibrate(timeline)
        out_json = args.out_dir / path.name.replace("_palette_timeline.json", "_calibrated_timeline.json")
        out_html = args.out_dir / path.name.replace("_palette_timeline.json", "_calibrated_timeline.html")
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        out_html.write_text(render_html(report), encoding="utf-8")
        for segment in report.get("segments", []):
            source_counts[segment["calibrated_source"]] += 1
            processing_counts[segment["calibrated_processing"]] += 1
        for region in report.get("regions", []):
            region_source_counts[region["calibrated_source"]] += 1
            region_processing_counts[region["calibrated_processing"]] += 1
            rows.append(
                {
                    "file": out_json.name,
                    "time": f"{region['start']:.2f}-{region['end']:.2f}s",
                    "source": region["calibrated_source"],
                    "source_evidence": ", ".join(region["source_evidence"][:4]),
                    "processing": region["calibrated_processing"],
                    "processing_evidence": ", ".join(region["processing_evidence"][:4]),
                }
            )

    summary = {
        "timeline_dir": str(args.timeline_dir),
        "source_counts": dict(source_counts.most_common()),
        "processing_counts": dict(processing_counts.most_common()),
        "region_source_counts": dict(region_source_counts.most_common()),
        "region_processing_counts": dict(region_processing_counts.most_common()),
        "rows": rows,
    }
    summary_json = args.out_dir / "calibrated_timeline_batch_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_rows = "\n".join(
        "<tr>"
        f"<td>{row['file']}</td><td>{row['time']}</td><td>{row['source']}</td><td>{row['source_evidence']}</td>"
        f"<td>{row['processing']}</td><td>{row['processing_evidence']}</td>"
        "</tr>"
        for row in rows
    )
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Calibrated Timeline Batch Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>Calibrated Timeline Batch Summary</h1>",
            "<div class='grid'>",
            counter_table("Segment Source Macro", source_counts),
            counter_table("Segment Processing Macro", processing_counts),
            counter_table("Region Source Macro", region_source_counts),
            counter_table("Region Processing Macro", region_processing_counts),
            "</div>",
            "<h2>Regions</h2>",
            "<table><thead><tr><th>Timeline</th><th>Time</th><th>Source Macro</th><th>Source Evidence</th><th>Processing Macro</th><th>Processing Evidence</th></tr></thead><tbody>",
            detail_rows,
            "</tbody></table>",
        ]
    )
    summary_html = args.out_dir / "calibrated_timeline_batch_summary.html"
    summary_html.write_text(html, encoding="utf-8")
    print(f"summary: {summary_json}")
    print(f"summary: {summary_html}")


if __name__ == "__main__":
    main()
