from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


def split_pipe(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split("|") if item.strip()}


def segment_key(row: dict) -> tuple[str, str, str]:
    return (Path(row["file"]).name, str(float(row["start"])), str(float(row["end"])))


def confidence_label(row: dict) -> str:
    support = row.get("teacher_support", "")
    our_groups = split_pipe(row.get("detected_groups", ""))
    teacher_groups = split_pipe(row.get("audioset_groups", ""))
    if support == "support":
        return "public_supported"
    if support == "disagree_or_missing" and our_groups and teacher_groups:
        return "public_disagrees"
    if support == "no_clear_teacher_group":
        return "teacher_unclear"
    return "not_checked"


def filter_groups(row: dict) -> tuple[str, str]:
    our_groups = split_pipe(row.get("detected_groups", ""))
    teacher_groups = split_pipe(row.get("audioset_groups", ""))
    support = row.get("teacher_support", "")
    if support == "support":
        kept = our_groups & teacher_groups
        return "|".join(sorted(kept or our_groups)), ""
    if support == "disagree_or_missing" and teacher_groups:
        return "", "|".join(sorted(our_groups))
    return "|".join(sorted(our_groups)), ""


def write_html(rows: list[dict], path: Path) -> None:
    trs = []
    for row in rows:
        clip = row.get("clip", "")
        audio_html = f'<audio controls preload="metadata" src="{html.escape(clip)}"></audio>' if clip else "-"
        trs.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td>{audio_html}</td>"
            f"<td>{html.escape(row.get('detected_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
            f"<td>{html.escape(row.get('audioset_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('audioset_top', '') or '-')}</td>"
            f"<td>{html.escape(row.get('teacher_confidence', '') or '-')}</td>"
            f"<td>{html.escape(row.get('teacher_filtered_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('teacher_suppressed_groups', '') or '-')}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Public Model Filtered Review</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.note {{ color: #444; }}
</style>
<h1>Public Model Filtered Review</h1>
<p class="note">AudioSet/AST is used as a broad public-model panel. Unsupported detections are demoted, not treated as absolute mistakes.</p>
<table>
<tr><th>Segment</th><th>Audio</th><th>Our Groups</th><th>Our Labels</th><th>AudioSet Groups</th><th>AudioSet Top Tags</th><th>Decision</th><th>Kept Groups</th><th>Suppressed Groups</th></tr>
{''.join(trs)}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=Path("outputs/review_queue_detected/audioset_teacher.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/review_queue_detected/public_model_filtered.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/review_queue_detected/public_model_filtered.html"))
    args = parser.parse_args()

    with args.teacher.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        kept, suppressed = filter_groups(row)
        out = dict(row)
        out["teacher_confidence"] = confidence_label(row)
        out["teacher_filtered_groups"] = kept
        out["teacher_suppressed_groups"] = suppressed
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    write_html(out_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
