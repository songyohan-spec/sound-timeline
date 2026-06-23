from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_items(items: list[dict], top_k: int, include_family: bool = False) -> str:
    rows = []
    for item in items[:top_k]:
        family_cell = f"<td>{item.get('family', '')}</td>" if include_family else ""
        rows.append(
            "<tr>"
            f"{family_cell}<td>{item['label']}</td><td>{float(item['score']):.8f}</td><td>{item.get('prompt', '')}</td>"
            "</tr>"
        )
    family_head = "<th>Family</th>" if include_family else ""
    return (
        "<table><thead><tr>"
        f"{family_head}<th>Label</th><th>Score</th><th>Winning Prompt</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    blocks = [
        "<h2>Top Overall</h2>",
        render_items(report["top_overall"], args.top_k, include_family=True),
    ]
    for family, items in report["families"].items():
        blocks.append(f"<h2>{family}</h2>")
        blocks.append(render_items(items, args.top_k, include_family=False))

    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Sound Palette CLAP Report</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.note{color:#555}</style>",
            "<h1>Sound Palette CLAP Report</h1>",
            f"<p><strong>Audio:</strong> {report['audio']}</p>",
            f"<p><strong>Model:</strong> {report['model']}</p>",
            "<p class='note'>This is an open-vocabulary sound-palette retrieval report. Read rankings as semantic hints.</p>",
            *blocks,
            f"<p class='note'>{report.get('caution', '')}</p>",
        ]
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

