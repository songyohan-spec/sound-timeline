from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_group(group: str, items: list[dict], top_k: int) -> str:
    rows = []
    for item in items[:top_k]:
        rows.append(f"<tr><td>{item['label']}</td><td>{float(item['score']):.6f}</td></tr>")
    return (
        f"<h2>{group}</h2>"
        "<table><thead><tr><th>Label</th><th>CLAP score</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    groups = report["scores"]
    blocks = [render_group(group, items, args.top_k) for group, items in groups.items()]
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>CLAP Semantic Scores</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f2f2f2}.note{color:#555}</style>",
            "<h1>CLAP Semantic Scores</h1>",
            f"<p><strong>Audio:</strong> {report['audio']}</p>",
            f"<p><strong>Model:</strong> {report['model']}</p>",
            "<p class='note'>Scores are normalized across all prompts in this prompt set. Use rankings as semantic hints, not probabilities.</p>",
            *blocks,
            f"<p class='note'>{report.get('caution', '')}</p>",
        ]
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

