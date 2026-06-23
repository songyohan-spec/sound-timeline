from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter, defaultdict
from pathlib import Path


FAILURE_ACTIONS = {
    "too_broad": "candidate masks are probably too wide; tighten regions or require stronger 4s support",
    "silent_selection": "selected stem/region is nearly silent; suppress from audition queues and inspect source-kind gating",
    "empty_candidate": "candidate is much quieter than context; suppress and inspect region/stem mismatch",
    "too_sparse": "candidate exists but is too sparse; keep only if transient synth hits are expected",
    "likely_overcaptures": "candidate may include most of the stem; compare residual before trusting it",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_cell(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def audio_src(path_value: str, html_path: Path) -> str:
    return Path(os.path.relpath(path_value, html_path.parent)).as_posix()


def failure_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if row.get("triage") != "auditionable"]


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "track",
        "triage",
        "labels",
        "stems",
        "candidate_rms_share",
        "candidate_rms_db",
        "residual_rms_db",
        "action",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict], all_rows: list[dict], out_html: Path, source_path: Path) -> None:
    status_counts = Counter(row.get("triage", "") for row in all_rows)
    fail_status_counts = Counter(row.get("triage", "") for row in rows)
    fail_label_counts = Counter()
    fail_stem_counts = Counter()
    label_by_status: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        labels = split_cell(row.get("labels", ""))
        stems = split_cell(row.get("stems", ""))
        fail_label_counts.update(labels)
        fail_stem_counts.update(stems)
        for label in labels:
            label_by_status[row.get("triage", "")][label] += 1

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    status_sections = []
    for status, counter in sorted(label_by_status.items()):
        status_sections.append(count_table(f"Labels in {status}", counter))

    trs = []
    for row in sorted(rows, key=lambda r: (r.get("triage", ""), -safe_float(r.get("candidate_rms_share")), r.get("track", ""))):
        mix_src = audio_src(row["stem_mix"], out_html)
        synth_src = audio_src(row["synth_candidate"], out_html)
        residual_src = audio_src(row["residual_context"], out_html)
        triage = row.get("triage", "")
        trs.append(
            "<tr>"
            f"<td><b>{html.escape(row['track'])}</b><br><small>{html.escape(row.get('stems', ''))}</small></td>"
            f"<td class='{html.escape(triage)}'><b>{html.escape(triage)}</b><br><small>{html.escape(FAILURE_ACTIONS.get(triage, 'inspect manually'))}</small></td>"
            f"<td>{html.escape(row.get('candidate_rms_share', ''))}</td>"
            f"<td>{html.escape(row.get('candidate_rms_db', ''))}</td>"
            f"<td>{html.escape(row.get('residual_rms_db', ''))}</td>"
            f"<td>{html.escape(row.get('labels', ''))}</td>"
            f"<td><audio controls src='{html.escape(mix_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(synth_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(residual_src)}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Candidate Failure Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
small {{ color: #555; }}
.too_broad {{ background: #fff2c8; }}
.silent_selection, .empty_candidate, .too_sparse, .likely_overcaptures {{ background: #f7dede; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Synth Candidate Failure Audit</h1>
<p class="note">Failure-only view filtered from <code>{html.escape(str(source_path))}</code>. This is for model and masking improvement, not for user-facing claims.</p>
<p>Failed rows: {len(rows)} / {len(all_rows)}</p>
{count_table("All Triage Status", status_counts)}
{count_table("Failure Status", fail_status_counts)}
{count_table("Failure Labels", fail_label_counts)}
{count_table("Failure Stems", fail_stem_counts)}
{''.join(status_sections)}
<section>
<h2>Failure Queue</h2>
<table>
<tr><th>Track</th><th>Failure</th><th>Candidate/Mix RMS</th><th>Candidate dB</th><th>Residual dB</th><th>Labels</th><th>Stem Mix</th><th>Synth Candidate</th><th>Residual Context</th></tr>
{''.join(trs)}
</table>
</section>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triage", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_failure_audit.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_failure_audit.html"))
    args = parser.parse_args()

    all_rows = read_rows(args.triage)
    rows = failure_rows(all_rows)
    for row in rows:
        row["action"] = FAILURE_ACTIONS.get(row.get("triage", ""), "inspect manually")
    write_csv(rows, args.out_csv)
    write_html(rows, all_rows, args.out_html, args.triage)
    print(f"failed: {len(rows)} / {len(all_rows)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
