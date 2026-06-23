from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_cell(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def audio_src(path_value: str, html_path: Path) -> str:
    return Path(os.path.relpath(path_value, html_path.parent)).as_posix()


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_table(title: str, counter: Counter[str]) -> str:
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
    return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"


def write_html(rows: list[dict], out_html: Path, source_path: Path) -> None:
    label_counts = Counter()
    stem_counts = Counter()
    for row in rows:
        label_counts.update(split_cell(row.get("labels", "")))
        stem_counts.update(split_cell(row.get("stems", "")))

    trs = []
    for index, row in enumerate(sorted(rows, key=lambda r: (-safe_float(r.get("candidate_rms_share")), r.get("track", ""))), start=1):
        mix_src = audio_src(row["stem_mix"], out_html)
        synth_src = audio_src(row["synth_candidate"], out_html)
        residual_src = audio_src(row["residual_context"], out_html)
        trs.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><b>{html.escape(row['track'])}</b><br><small>{html.escape(row.get('stems', ''))}</small></td>"
            f"<td>{html.escape(row.get('candidate_rms_share', ''))}</td>"
            f"<td>{html.escape(row.get('labels', ''))}</td>"
            f"<td><audio controls src='{html.escape(synth_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(mix_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(residual_src)}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Final Synth Audition Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 250px; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Final Synth Audition Queue</h1>
<p class="note">Filtered from <code>{html.escape(str(source_path))}</code>. This page contains only candidates marked <b>reliable</b> after strict synth export, separation triage, and second-pass reliability filtering. It is still pseudo-separation, not ground truth.</p>
<p>Rows: {len(rows)}</p>
{count_table("Labels", label_counts)}
{count_table("Stems", stem_counts)}
<section>
<h2>Listen Queue</h2>
<table>
<tr><th>#</th><th>Track</th><th>Candidate/Mix RMS</th><th>Labels</th><th>Synth Candidate</th><th>Stem Mix</th><th>Residual Context</th></tr>
{''.join(trs)}
</table>
</section>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reliable", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_reliable.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_final.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_final.html"))
    args = parser.parse_args()

    rows = [row for row in read_rows(args.reliable) if row.get("reliability") == "reliable"]
    if not rows:
        raise SystemExit("No reliable rows to render.")
    write_csv(rows, args.out_csv)
    write_html(rows, args.out_html, args.reliable)
    print(f"selected: {len(rows)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
