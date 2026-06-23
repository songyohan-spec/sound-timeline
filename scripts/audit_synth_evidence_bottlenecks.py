from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


FOCUS_LABELS = {
    "synth_pad_wash",
    "supersaw_stack",
    "digital_synth_lead",
    "bitcrushed_synth_lead",
    "arpeggio_sequence",
    "vocal_synth_hybrid",
    "formant_vocoder",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_html(summary_rows: list[dict], detail_rows: list[dict], out_html: Path) -> None:
    def table(headers: list[str], rows: list[list[str]]) -> str:
        head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
            for row in rows
        )
        return f"<table><tr>{head}</tr>{body}</table>"

    summary = table(
        ["Label", "Total", "Usable", "Review", "Ignored", "Main Support", "Main Decisions"],
        [
            [
                row["label"],
                row["total"],
                row["usable"],
                row["review"],
                row["ignored"],
                row["support_counts"],
                row["decision_counts"],
            ]
            for row in summary_rows
        ],
    )
    details = table(
        ["Track", "Stem", "Time", "Label", "Conf", "Support", "Decision", "Matches"],
        [
            [
                row["track"],
                row["stem"],
                f"{row['start']}-{row['end']}s",
                row["specialist_label"],
                row["specialist_conf"],
                row["source_kind_support"],
                row["decision"],
                row["support_matches"],
            ]
            for row in detail_rows
        ],
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Evidence Bottleneck Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Synth Evidence Bottleneck Audit</h1>
<p class="note">Focuses on labels that should matter for synth reading but are currently underrepresented or fragile. This is for deciding what evidence/modeling to improve next.</p>
<h2>Summary</h2>
{summary}
<h2>Details</h2>
{details}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v3_dsp.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_evidence_bottlenecks.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_evidence_bottlenecks.html"))
    args = parser.parse_args()

    rows = [row for row in read_rows(args.input) if row["specialist_label"] in FOCUS_LABELS]
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["specialist_label"]].append(row)

    summary_rows = []
    for label in sorted(FOCUS_LABELS):
        label_rows = by_label.get(label, [])
        decisions = Counter(row["decision"] for row in label_rows)
        supports = Counter(row["source_kind_support"] for row in label_rows)
        usable = sum(1 for row in label_rows if row["final_label"] != "ambiguous")
        review = decisions["needs_review_or_more_data"] + decisions["ambiguous_family_only"]
        ignored = decisions["ignore"]
        summary_rows.append(
            {
                "label": label,
                "total": len(label_rows),
                "usable": usable,
                "review": review,
                "ignored": ignored,
                "support_counts": "|".join(f"{k}:{v}" for k, v in supports.most_common()),
                "decision_counts": "|".join(f"{k}:{v}" for k, v in decisions.most_common()),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    detail_rows = sorted(
        rows,
        key=lambda row: (
            row["specialist_label"],
            row["decision"] != "needs_review_or_more_data",
            row["decision"] != "ignore",
            -float(row.get("specialist_conf") or 0.0),
        ),
    )[:300]
    write_html(summary_rows, detail_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
