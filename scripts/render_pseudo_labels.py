from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No rows found.")

    visible = [
        "start",
        "end",
        "source_primary",
        "source_primary_conf",
        "source_alt",
        "source_ambiguous",
        "has_long_tail",
        "feels_wide",
        "feels_rough_or_crushed",
        "has_filtering",
        "gets_brighter_or_filter_moves",
        "has_pumping_or_motion",
        "needs_review",
        "low_confidence_fields",
        "dsp_brightness",
        "dsp_flatness",
        "dsp_stereo_side_ratio",
        "dsp_motion_strength",
        "dsp_feels_wide",
        "dsp_has_motion",
        "dsp_feels_rough_or_noisy",
        "dsp_feels_bright",
        "agreement_wide",
        "agreement_motion",
        "agreement_rough",
    ]
    headers = [key for key in visible if key in rows[0]]
    table_rows = []
    for row in rows:
        cells = []
        for key in headers:
            value = row.get(key, "")
            cls = ""
            if key == "needs_review" and value == "yes":
                cls = " class='warn'"
            cells.append(f"<td{cls}>{value}</td>")
        table_rows.append("<tr>" + "".join(cells) + "</tr>")

    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Pseudo Labels</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1280px;margin:28px auto;color:#161616}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}th{background:#f2f2f2}.warn{background:#fff0d6;font-weight:bold}.note{color:#555}</style>",
            "<h1>Pseudo Labels</h1>",
            "<p class='note'>These are automatic weak labels from the current model plus DSP cues. They are not ground truth.</p>",
            "<table>",
            "<thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead>",
            "<tbody>",
            *table_rows,
            "</tbody></table>",
        ]
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
