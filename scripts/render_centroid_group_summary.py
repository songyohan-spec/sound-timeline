import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row["source_csv"] = str(path)
                row["start"] = float(row["start"])
                row["end"] = float(row["end"])
                row["confidence"] = float(row["confidence"])
                rows.append(row)
    return rows


def write_html(rows, out, title):
    by_audio = defaultdict(list)
    for row in rows:
        by_audio[row["audio"]].append(row)

    overall = Counter(row["label"] for row in rows)
    overall_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in overall.most_common()
    )

    sections = []
    for audio, audio_rows in sorted(by_audio.items()):
        counts = Counter(row["label"] for row in audio_rows)
        dominant, dominant_count = counts.most_common(1)[0]
        mean_conf = sum(row["confidence"] for row in audio_rows) / len(audio_rows)
        timeline = "".join(
            "<tr>"
            f"<td>{row['start']:.2f}-{row['end']:.2f}s</td>"
            f"<td>{html.escape(row['label'])}</td>"
            f"<td>{row['confidence']:.3f}</td>"
            f"<td>{html.escape(row.get('alternatives', ''))}</td>"
            "</tr>"
            for row in audio_rows
        )
        count_rows = "".join(
            f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
            for label, count in counts.most_common()
        )
        sections.append(
            f"<h2>{html.escape(audio)}</h2>"
            f"<p>Dominant: <strong>{html.escape(dominant)}</strong> ({dominant_count}/{len(audio_rows)} segments), "
            f"mean top confidence: <strong>{mean_conf:.3f}</strong></p>"
            "<h3>Label Counts</h3>"
            f"<table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{count_rows}</tbody></table>"
            "<h3>Timeline</h3>"
            "<table><thead><tr><th>Time</th><th>Top Label</th><th>Confidence</th><th>Alternatives</th></tr></thead>"
            f"<tbody>{timeline}</tbody></table>"
        )

    style = """
    body{font-family:Arial,sans-serif;max-width:1150px;margin:32px auto;color:#111}
    table{border-collapse:collapse;width:100%;margin:10px 0 24px;font-size:13px}
    th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}
    th{background:#eee}
    """
    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{style}</style></head>
<body>
<h1>{html.escape(title)}</h1>
<h2>Overall Label Counts</h2>
<table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{overall_rows}</tbody></table>
{''.join(sections)}
</body>
</html>
"""
    Path(out).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out-html", required=True)
    parser.add_argument("--title", default="Centroid Timeline Group Summary")
    args = parser.parse_args()
    rows = read_rows(args.inputs)
    write_html(rows, args.out_html, args.title)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
