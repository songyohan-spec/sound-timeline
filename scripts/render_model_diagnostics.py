from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


FAMILIES = ["vocals", "synth", "guitar_strings", "bass", "drums", "noise_fx", "sampled_loop"]


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def float_value(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row: dict) -> tuple[str, str, str]:
    return (row["file"], row["start"], row["end"])


def top_items(counter: Counter[str], limit: int = 12) -> str:
    if not counter:
        return "<tr><td>-</td><td>0</td></tr>"
    return "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in counter.most_common(limit)
    )


def write_html(
    matrix_rows: list[dict],
    timeline_rows: list[dict],
    filtered_rows: list[dict],
    out_html: Path,
) -> None:
    timeline_by_key = {key(row): row for row in timeline_rows}
    filtered_by_key = {key(row): row for row in filtered_rows}

    confidence_counter = Counter(row.get("confidence", "") for row in timeline_rows)
    primary_counter = Counter(row.get("primary_read", "") for row in timeline_rows)
    active_counter: Counter[str] = Counter()
    weak_files: Counter[str] = Counter()
    disputed_files: Counter[str] = Counter()
    demoted_counter: Counter[str] = Counter()
    dense_rows = []
    empty_rows = []
    top_disputed = []

    for row in matrix_rows:
        active = split_pipe(row.get("active_layers", ""))
        active_counter.update(active)
        trow = timeline_by_key.get(key(row), {})
        frow = filtered_by_key.get(key(row), {})
        if trow.get("confidence") in {"weak", "disputed"}:
            weak_files[row["file"]] += 1
        if trow.get("confidence") == "disputed":
            disputed_files[row["file"]] += 1
        demoted = split_pipe(frow.get("teacher_suppressed_groups", ""))
        demoted_counter.update(demoted)
        if len(active) >= 4:
            dense_rows.append((row, active))
        if not active:
            empty_rows.append(row)
        if trow.get("confidence") == "disputed":
            top_disputed.append((row, trow, frow))

    def table(title: str, counter: Counter[str]) -> str:
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{top_items(counter)}</table></section>"

    def segment_rows(rows: list, limit: int = 20) -> str:
        body = []
        for item in rows[:limit]:
            if len(item) == 3:
                row, trow, frow = item
                detail = (
                    f"primary={trow.get('primary_read', '-')}; "
                    f"demoted={frow.get('teacher_suppressed_groups', '-') or '-'}; "
                    f"audioset={frow.get('audioset_groups', '-') or '-'}"
                )
            else:
                row = item[0] if isinstance(item, tuple) else item
                detail = f"active={row.get('active_layers', '-') or '-'}"
            body.append(
                "<tr>"
                f"<td>{html.escape(row['file'])}</td>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td>{html.escape(detail)}</td>"
                "</tr>"
            )
        return "".join(body) or "<tr><td>-</td><td>-</td><td>-</td></tr>"

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Model Diagnostics</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(220px, 1fr)); gap: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; }}
</style>
<h1>Model Diagnostics</h1>
<p class="note">Use this to decide what to improve next. It highlights broad repeated behavior, not individual truth.</p>
<div class="grid">
{table("Primary Reads", primary_counter)}
{table("Confidence", confidence_counter)}
{table("Active Layers", active_counter)}
{table("Weak / Disputed By File", weak_files)}
{table("Disputed By File", disputed_files)}
{table("Demoted Groups", demoted_counter)}
</div>
<h2>Most Crowded Segments</h2>
<table><tr><th>File</th><th>Time</th><th>Detail</th></tr>{segment_rows(dense_rows)}</table>
<h2>Empty / No Active Layer Segments</h2>
<table><tr><th>File</th><th>Time</th><th>Detail</th></tr>{segment_rows(empty_rows)}</table>
<h2>Disputed Segments</h2>
<table><tr><th>File</th><th>Time</th><th>Detail</th></tr>{segment_rows(top_disputed)}</table>
<h2>Next Modeling Implication</h2>
<ul>
<li>If a group is frequently demoted, lower its project-model influence or add hard negatives.</li>
<li>If many segments are crowded, raise active-layer thresholds or split broad cues into more specific cues.</li>
<li>If many segments are weak, use public cues as hints only and avoid definitive wording.</li>
</ul>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_layer_matrix.csv"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_element_timeline.csv"))
    parser.add_argument("--filtered", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/public_model_filtered.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/model_diagnostics.html"))
    args = parser.parse_args()
    write_html(load_csv(args.matrix), load_csv(args.timeline), load_csv(args.filtered), args.out_html)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
