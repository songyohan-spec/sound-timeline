from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(name).stem)


def top_clap(path: Path, group: str) -> str:
    report = json.loads(path.read_text(encoding="utf-8"))
    values = report.get("scores", {}).get(group, [])
    return values[0]["label"] if values else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rf-summary", type=Path, default=Path("outputs/external_batch_alt2/summary.csv"))
    parser.add_argument("--clap-dir", type=Path, default=Path("outputs/external_clap"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/external_clap/ensemble_summary.html"))
    args = parser.parse_args()

    with args.rf_summary.open("r", encoding="utf-8", newline="") as f:
        rf_rows = list(csv.DictReader(f))
    if not rf_rows:
        raise SystemExit("No RF summary rows found.")

    rows = []
    clap_source_counts: Counter[str] = Counter()
    clap_space_counts: Counter[str] = Counter()
    clap_texture_counts: Counter[str] = Counter()
    clap_motion_counts: Counter[str] = Counter()

    for rf in rf_rows:
        clap_json = args.clap_dir / f"{safe_stem(rf['file'])}_clap.json"
        if not clap_json.exists():
            continue
        clap_source = top_clap(clap_json, "source")
        clap_space = top_clap(clap_json, "space")
        clap_texture = top_clap(clap_json, "texture")
        clap_motion = top_clap(clap_json, "motion")
        clap_source_counts[clap_source] += 1
        clap_space_counts[clap_space] += 1
        clap_texture_counts[clap_texture] += 1
        clap_motion_counts[clap_motion] += 1
        rows.append(
            {
                "file": rf["file"],
                "rf_source": rf["source"],
                "clap_source": clap_source,
                "rf_reverb": rf["reverb"],
                "clap_space": clap_space,
                "rf_distortion": rf["distortion"],
                "clap_texture": clap_texture,
                "rf_stereo": rf["stereo"],
                "rf_motion": rf["motion"],
                "clap_motion": clap_motion,
                "needs_review": rf["needs_review"],
            }
        )

    def counter_table(title: str, counter: Counter[str]) -> str:
        body = "\n".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in counter.most_common())
        return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"

    detail = []
    for row in rows:
        detail.append(
            "<tr>"
            f"<td>{row['file']}</td><td>{row['rf_source']}</td><td>{row['clap_source']}</td>"
            f"<td>{row['rf_reverb']}</td><td>{row['clap_space']}</td>"
            f"<td>{row['rf_distortion']}</td><td>{row['clap_texture']}</td>"
            f"<td>{row['rf_stereo']}</td><td>{row['rf_motion']}</td><td>{row['clap_motion']}</td>"
            f"<td>{row['needs_review']}</td>"
            "</tr>"
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Batch Ensemble Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1280px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>Batch Ensemble Summary</h1>",
            f"<p>Matched clips: <strong>{len(rows)}</strong></p>",
            "<div class='grid'>",
            counter_table("CLAP Source", clap_source_counts),
            counter_table("CLAP Space", clap_space_counts),
            counter_table("CLAP Texture", clap_texture_counts),
            counter_table("CLAP Motion", clap_motion_counts),
            "</div>",
            "<h2>RF/DSP vs CLAP Details</h2>",
            "<table><thead><tr><th>File</th><th>RF Source</th><th>CLAP Source</th><th>RF Reverb</th><th>CLAP Space</th><th>RF Distortion</th><th>CLAP Texture</th><th>RF Stereo</th><th>RF Motion</th><th>CLAP Motion</th><th>Review</th></tr></thead><tbody>",
            *detail,
            "</tbody></table>",
        ]
    )
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(html, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

