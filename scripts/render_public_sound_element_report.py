from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def parse_top_labels(value: str, limit: int = 6) -> list[str]:
    labels = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, rest = part.rsplit(":", 1)
        labels.append(f"{label.strip()} ({rest.strip()})")
    return labels[:limit]


def compact_audioset(value: str, limit: int = 6) -> str:
    tags = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if part:
            tags.append(part)
    return "<br>".join(html.escape(tag) for tag in tags[:limit]) or "-"


def decision(row: dict) -> tuple[str, str]:
    status = row.get("teacher_confidence", "")
    if status == "public_supported":
        kept = row.get("teacher_filtered_groups", "")
        return "supported", f"Public AudioSet panel supports: {kept or 'detected cue'}."
    if status == "public_disagrees":
        suppressed = row.get("teacher_suppressed_groups", "")
        groups = row.get("audioset_groups", "")
        return "demoted", f"Our {suppressed or 'cue'} detection is not supported; public panel leans {groups or 'elsewhere'}."
    if status == "teacher_unclear":
        return "unclear", "Public panel did not produce a clear broad category."
    return "unchecked", "No public-model check was available."


def write_html(rows: list[dict], out_html: Path) -> None:
    counters = {
        "supported_groups": Counter(),
        "demoted_groups": Counter(),
        "audioset_groups": Counter(),
        "labels": Counter(),
    }
    for row in rows:
        for group in split_pipe(row.get("teacher_filtered_groups", "")):
            counters["supported_groups"][group] += 1
        for group in split_pipe(row.get("teacher_suppressed_groups", "")):
            counters["demoted_groups"][group] += 1
        for group in split_pipe(row.get("audioset_groups", "")):
            counters["audioset_groups"][group] += 1
        for label in split_pipe(row.get("detected_labels", "")):
            counters["labels"][label] += 1

    def table(title: str, counter: Counter[str]) -> str:
        body = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
            for label, count in counter.most_common()
        )
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body or '<tr><td>-</td><td>0</td></tr>'}</table></section>"

    detail_rows = []
    for idx, row in enumerate(rows, 1):
        badge, text = decision(row)
        candidates = "<br>".join(html.escape(item) for item in parse_top_labels(row.get("top_labels", ""))) or "-"
        detail_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row.get('clip', ''))}\"></audio></td>"
            f"<td><span class=\"badge {html.escape(badge)}\">{html.escape(badge)}</span><br>{html.escape(text)}</td>"
            f"<td>{html.escape(row.get('teacher_filtered_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('teacher_suppressed_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
            f"<td>{html.escape(row.get('audioset_groups', '') or '-')}</td>"
            f"<td>{compact_audioset(row.get('audioset_top', ''))}</td>"
            f"<td>{candidates}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Open-Source Panel Sound Element Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 18px; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 12px; }}
.supported {{ background: #dff3df; }}
.demoted {{ background: #ffe1df; }}
.unclear {{ background: #fff2c6; }}
.unchecked {{ background: #e7e7e7; }}
.note {{ color: #444; }}
</style>
<h1>Open-Source Panel Sound Element Report</h1>
<p class="note">This report combines the project model with a public AudioSet/AST teacher panel. Demoted labels are not deleted forever; they are downgraded because a broad public model did not support them.</p>
<div class="grid">
{table("Public-Supported Groups", counters["supported_groups"])}
{table("Public-Demoted Groups", counters["demoted_groups"])}
{table("AudioSet Broad Reads", counters["audioset_groups"])}
{table("Project Label Mentions", counters["labels"])}
</div>
<h2>Segment Detail</h2>
<table>
<tr><th>#</th><th>Segment</th><th>Audio</th><th>Decision</th><th>Kept Groups</th><th>Demoted Groups</th><th>Project Labels</th><th>AudioSet Groups</th><th>AudioSet Top Tags</th><th>Project Candidates</th></tr>
{''.join(detail_rows)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/review_queue_detected/public_model_filtered.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/review_queue_detected/open_source_panel_report.html"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
