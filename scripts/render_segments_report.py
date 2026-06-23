from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "source_family": "Source",
    "reverb": "Reverb",
    "distortion": "Distortion",
    "filter_presence": "Filter",
    "filter_motion_type": "Filter Motion",
    "stereo": "Stereo",
    "motion_presence": "Motion",
}


def format_label(label: str) -> str:
    return label.replace("_", " ")


def top_label(segment: dict, key: str) -> str:
    values = segment["predictions"].get(key)
    if not values:
        return "-"
    top = values[0]
    top_conf = float(top["confidence"])
    top_text = f"{format_label(top['label'])} {top_conf * 100:.0f}%"

    if top_conf >= 0.75:
        return top_text

    if len(values) > 1:
        alt = values[1]
        alt_conf = float(alt["confidence"])
        alt_text = f"{format_label(alt['label'])} {alt_conf * 100:.0f}%"
        if top_conf < 0.55 or (top_conf - alt_conf) < 0.15:
            return f"<span class='ambiguous'>ambiguous</span><br>{top_text}<br>{alt_text}"
        return f"<span class='possible'>possible</span><br>{top_text}<br>{alt_text}"

    if top_conf < 0.55:
        return f"<span class='ambiguous'>ambiguous</span><br>{top_text}"
    return f"<span class='possible'>possible</span><br>{top_text}"


def render_html(report: dict) -> str:
    keys = ["source_family", "reverb", "distortion", "filter_presence", "filter_motion_type", "stereo", "motion_presence"]
    rows = []
    for segment in report["segments"]:
        cells = [
            f"{segment['start']:.1f}-{segment['end']:.1f}s",
            *[top_label(segment, key) for key in keys if key in segment["predictions"]],
        ]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    headers = ["Time"] + [DISPLAY_NAMES.get(key, key) for key in keys if report["segments"] and key in report["segments"][0]["predictions"]]
    return "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Segment Sound Profile</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1180px;margin:28px auto;color:#161616}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f1f1f1}tr:nth-child(even){background:#fafafa}.note{margin-top:18px;color:#555}.ambiguous{display:inline-block;color:#8a4b00;font-weight:bold;margin-bottom:3px}.possible{display:inline-block;color:#6a5d00;font-weight:bold;margin-bottom:3px}</style>",
            "<h1>Segment Sound Profile</h1>",
            f"<p><strong>Audio:</strong> {report['audio']}</p>",
            "<p>Predictions below 75% are softened as <strong>possible</strong>; low or close top-2 predictions are marked <strong>ambiguous</strong>.</p>",
            "<table>",
            "<thead><tr>" + "".join(f"<th>{header}</th>" for header in headers) + "</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table>",
            f"<p class='note'>{report.get('caution', '')}</p>",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(render_html(report), encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
