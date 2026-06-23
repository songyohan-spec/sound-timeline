from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_cell(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def count_table(title: str, counter: Counter[str], limit: int = 12) -> str:
    if not counter:
        return f"<section><h2>{html.escape(title)}</h2><p class='muted'>No rows.</p></section>"
    body = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in counter.most_common(limit)
    )
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"
    )


def summarize_rows(rows: list[dict[str, str]]) -> tuple[Counter[str], Counter[str], Counter[str]]:
    label_counts = Counter()
    stem_counts = Counter()
    track_counts = Counter()
    for row in rows:
        label_counts.update(split_cell(row.get("labels", "")))
        stem_counts.update(split_cell(row.get("stems", "")))
        track = row.get("track")
        if track:
            track_counts[track] += 1
    return label_counts, stem_counts, track_counts


def write_html(
    out_html: Path,
    triage_rows: list[dict[str, str]],
    auditionable_rows: list[dict[str, str]],
    reliable_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    failure_rows: list[dict[str, str]],
) -> None:
    final_labels, final_stems, final_tracks = summarize_rows(final_rows)
    reliable_labels, _, _ = summarize_rows(reliable_rows)
    failure_reasons = Counter(
        row.get("failure_reason") or row.get("action") or "unknown"
        for row in failure_rows
    )
    triage_counts = Counter(row.get("triage", "unknown") or "unknown" for row in triage_rows)
    reliability_counts = Counter(row.get("reliability", "unknown") or "unknown" for row in reliable_rows)

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Current Best Synth Pipeline Status</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; max-width: 980px; }}
.card {{ border: 1px solid #d0d0d0; padding: 12px; background: #fafafa; }}
.big {{ font-size: 28px; font-weight: 700; }}
.muted {{ color: #555; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
</style>
<h1>Current Best Synth Pipeline Status</h1>
<p class="muted">This is a build status page, not a model claim. It shows how the current pseudo-separation pipeline filters synth candidates from Demucs stems.</p>
<section>
<h2>Funnel</h2>
<div class="grid">
<div class="card"><div class="big">{len(triage_rows)}</div><div>strict exported candidates</div></div>
<div class="card"><div class="big">{len(auditionable_rows)}</div><div>auditionable candidates</div></div>
<div class="card"><div class="big">{len(reliable_rows)}</div><div>reliability-scored candidates</div></div>
<div class="card"><div class="big">{len(final_rows)}</div><div>final listen queue</div></div>
</div>
</section>
{count_table("Final Labels", final_labels)}
{count_table("Final Stems", final_stems)}
{count_table("Final Tracks", final_tracks)}
{count_table("Reliable Labels Before Final Filter", reliable_labels)}
{count_table("Strict Triage", triage_counts)}
{count_table("Reliable Queue Status", reliability_counts)}
{count_table("Failure Reasons", failure_reasons)}
<section>
<h2>How To Read This</h2>
<ul>
<li><b>Final listen queue</b> is the main page to audition first. It is intentionally smaller than the broad timeline.</li>
<li><b>Reliable</b> means the candidate passed current heuristic checks, not that it is a true isolated synth stem.</li>
<li>If a label dominates the final queue, that is a useful model bias signal. It may reflect the source collection, the music clips, or the current label ontology.</li>
<li>The next modeling step should use the final queue as a teacher/audit source, then add targeted negative examples for false synth detections.</li>
</ul>
</section>
</html>"""
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/demucs_stems_6s_full"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/current_best_status.html"))
    args = parser.parse_args()

    triage_rows = read_rows(args.root / "synth_candidate_audio_strict_triage.csv")
    auditionable_rows = read_rows(args.root / "synth_candidate_audio_auditionable.csv")
    reliable_rows = read_rows(args.root / "synth_candidate_audio_reliable.csv")
    final_rows = read_rows(args.root / "synth_candidate_audio_final.csv")
    failure_rows = read_rows(args.root / "synth_candidate_audio_failure_audit.csv")
    write_html(args.out_html, triage_rows, auditionable_rows, reliable_rows, final_rows, failure_rows)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
