from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/external_batch"))
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()
    out_csv = args.out_csv or (args.batch_dir / "summary.csv")
    out_html = args.out_html or (args.batch_dir / "summary.html")

    files = sorted(args.batch_dir.glob("*_pseudo_labels.csv"))
    if not files:
        raise SystemExit(f"No *_pseudo_labels.csv files found in {args.batch_dir}")

    rows = []
    source_counts: Counter[str] = Counter()
    reverb_counts: Counter[str] = Counter()
    distortion_counts: Counter[str] = Counter()
    stereo_counts: Counter[str] = Counter()
    motion_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    low_conf_fields: Counter[str] = Counter()
    agreement_counts: Counter[str] = Counter()
    confidences: dict[str, list[float]] = {
        "source": [],
        "reverb": [],
        "distortion": [],
        "stereo": [],
        "motion": [],
    }

    for path in files:
        for row in read_csv(path):
            item = {
                "file": path.name.replace("_pseudo_labels.csv", ""),
                "start": row.get("start", ""),
                "end": row.get("end", ""),
                "source": row.get("source_primary", ""),
                "source_conf": row.get("source_primary_conf", ""),
                "source_alt": row.get("source_alt", ""),
                "source_ambiguous": row.get("source_ambiguous", ""),
                "reverb": row.get("reverb_label", ""),
                "reverb_conf": row.get("reverb_conf", ""),
                "distortion": row.get("distortion_label", ""),
                "distortion_conf": row.get("distortion_conf", ""),
                "stereo": row.get("stereo_label", ""),
                "stereo_conf": row.get("stereo_conf", ""),
                "motion": row.get("motion_label", ""),
                "motion_conf": row.get("motion_conf", ""),
                "needs_review": row.get("needs_review", ""),
                "low_confidence_fields": row.get("low_confidence_fields", ""),
            }
            rows.append(item)
            source_counts[item["source"]] += 1
            reverb_counts[item["reverb"]] += 1
            distortion_counts[item["distortion"]] += 1
            stereo_counts[item["stereo"]] += 1
            motion_counts[item["motion"]] += 1
            review_counts[item["needs_review"]] += 1
            for field in item["low_confidence_fields"].split(";"):
                if field:
                    low_conf_fields[field] += 1
                    if field.startswith("dsp_disagreement:"):
                        for name in field.split(":", 1)[1].split(","):
                            if name:
                                agreement_counts[name] += 1
            for key, conf_key in [
                ("source", "source_conf"),
                ("reverb", "reverb_conf"),
                ("distortion", "distortion_conf"),
                ("stereo", "stereo_conf"),
                ("motion", "motion_conf"),
            ]:
                try:
                    confidences[key].append(float(item[conf_key]))
                except ValueError:
                    pass

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def counter_table(title: str, counter: Counter[str]) -> str:
        body = "\n".join(f"<tr><td>{key}</td><td>{value}</td></tr>" for key, value in counter.most_common())
        return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"

    def confidence_table() -> str:
        body = "\n".join(
            f"<tr><td>{key}</td><td>{mean(values):.3f}</td></tr>"
            for key, values in confidences.items()
            if values
        )
        return f"<h2>Average Confidence</h2><table><thead><tr><th>Field</th><th>Mean Top-1 Confidence</th></tr></thead><tbody>{body}</tbody></table>"

    detail_rows = []
    for row in rows:
        cls = " class='warn'" if row["needs_review"] == "yes" else ""
        detail_rows.append(
            "<tr>"
            f"<td>{row['file']}</td><td>{row['start']}-{row['end']}</td><td>{row['source']} ({row['source_conf']})</td>"
            f"<td>{row['reverb']} ({row['reverb_conf']})</td><td>{row['distortion']} ({row['distortion_conf']})</td>"
            f"<td>{row['stereo']} ({row['stereo_conf']})</td><td>{row['motion']} ({row['motion_conf']})</td>"
            f"<td{cls}>{row['needs_review']}</td><td>{row['low_confidence_fields']}</td>"
            "</tr>"
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>External Batch Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1280px;margin:28px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:7px;text-align:left;vertical-align:top}th{background:#f2f2f2}.warn{background:#fff0d6;font-weight:bold}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>External Batch Summary</h1>",
            f"<p>Total segments: <strong>{len(rows)}</strong></p>",
            f"<p>Needs review ratio: <strong>{review_counts.get('yes', 0)}/{len(rows)}</strong></p>",
            "<div class='grid'>",
            counter_table("Needs Review", review_counts),
            counter_table("Low Confidence Fields", low_conf_fields),
            counter_table("DSP Disagreements", agreement_counts),
            confidence_table(),
            counter_table("Source", source_counts),
            counter_table("Reverb", reverb_counts),
            counter_table("Distortion", distortion_counts),
            counter_table("Stereo", stereo_counts),
            counter_table("Motion", motion_counts),
            "</div>",
            "<h2>Details</h2>",
            "<table><thead><tr><th>File</th><th>Time</th><th>Source</th><th>Reverb</th><th>Distortion</th><th>Stereo</th><th>Motion</th><th>Review</th><th>Low Confidence</th></tr></thead><tbody>",
            *detail_rows,
            "</tbody></table>",
        ]
    )
    out_html.write_text(html, encoding="utf-8")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()
