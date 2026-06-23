from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def rel_audio(path: str, out_html: Path) -> str:
    raw = Path(path)
    try:
        return raw.relative_to(out_html.parent).as_posix()
    except ValueError:
        return raw.as_posix()


def count_table(title: str, counter: Counter[str]) -> str:
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
    return f"<h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_queue_v4_strict.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_queue_v4_strict.html"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    label_counts = Counter(row["synth_label_top"] for row in rows)
    decision_counts = Counter(row["ensemble_decision"] for row in rows)
    support_counts = Counter(row["source_kind_support"] for row in rows)
    detail = []
    for row in rows:
        detail.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload='none' src='{html.escape(rel_audio(row['stem_path'], args.out_html))}'></audio></td>"
            f"<td>{html.escape(row['synth_label_top'])} ({row['synth_label_conf']})</td>"
            f"<td>{html.escape(row['synth_label_alternatives'])}</td>"
            f"<td>{html.escape(row['ensemble_decision'])}</td>"
            f"<td>{html.escape(row['source_kind_support'])}</td>"
            f"<td>{html.escape(row['support_matches'])}</td>"
            f"<td>{row['priority_score']}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Teacher Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Synth Teacher Queue</h1>
<p class="note">Prioritized stem segments for external teacher checks or tiny human review. This is deliberately diverse by synth label, not a random sample.</p>
{count_table("Labels", label_counts)}
{count_table("Current Decisions", decision_counts)}
{count_table("Current Source-Kind Support", support_counts)}
<h2>Details</h2>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Audio</th><th>Specialist</th><th>Alternatives</th><th>Decision</th><th>Support</th><th>Matches</th><th>Priority</th></tr>
{''.join(detail)}
</table>
</html>"""
    args.out_html.write_text(page, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
