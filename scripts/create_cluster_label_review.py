import argparse
import csv
import html
import json
from pathlib import Path


def load_manifest(path):
    clips = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            clips.setdefault(int(row["cluster"]), []).append(row)
    return clips


def feature_read(features):
    bright = features.get("centroid", 0) > 0.6 or features.get("rolloff", 0) > 0.6
    dark = features.get("centroid", 0) < -0.5 or features.get("rolloff", 0) < -0.5
    noisy = features.get("flatness", 0) > 0.4 or features.get("zcr", 0) > 0.5
    wide = features.get("width", 0) > 0.5
    narrow = features.get("width", 0) < -0.45
    dynamic = features.get("rms_range", 0) > 0.45 or features.get("motion_strength", 0) > 0.45

    words = []
    if bright:
        words.append("bright")
    if dark:
        words.append("dark")
    if noisy:
        words.append("noisy/grainy")
    if wide:
        words.append("wide")
    if narrow:
        words.append("narrow/centered")
    if dynamic:
        words.append("dynamic/pulsing")
    return ", ".join(words) if words else "mid/ambiguous"


def write_html(rows, path):
    style = """
    body{font-family:Arial,sans-serif;margin:24px;color:#111}
    table{border-collapse:collapse;width:100%;font-size:13px}
    th,td{border:1px solid #ddd;padding:8px;vertical-align:top}
    th{background:#eee;text-align:left}
    .warn{background:#fff3cd}
    code{font-family:Consolas,monospace}
    """
    html_rows = []
    for row in rows:
        clips = "<br>".join(
            f"<code>{html.escape(c)}</code>" for c in row["representative_clips"].split(" | ") if c
        )
        html_rows.append(
            "<tr>"
            f"<td>{row['cluster']}</td>"
            f"<td>{row['count']}</td>"
            f"<td>{html.escape(row['auto_name'])}</td>"
            f"<td>{html.escape(row['feature_read'])}</td>"
            f"<td>{html.escape(row['distinctive_features'])}</td>"
            f"<td>{clips}</td>"
            f"<td class='warn'>{html.escape(row['human_label'])}</td>"
            f"<td>{html.escape(row['notes'])}</td>"
            "</tr>"
        )
    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Cluster Label Review</title><style>{style}</style></head>
<body>
<h1>Cluster Label Review</h1>
<p>Listen to the representative clips and name only these clusters. This replaces per-segment manual labeling.</p>
<table>
<thead>
<tr>
<th>Cluster</th><th>Count</th><th>Auto Name</th><th>DSP Read</th><th>Distinctive Features</th><th>Representative Clips</th><th>Human Label</th><th>Notes</th>
</tr>
</thead>
<tbody>{''.join(html_rows)}</tbody>
</table>
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    clusters = json.loads(Path(args.clusters_json).read_text(encoding="utf-8"))
    clips_by_cluster = load_manifest(args.manifest)
    rows = []

    for cluster in clusters:
        cid = int(cluster["cluster"])
        distinctive = cluster.get("distinctive_features", {})
        clips = [row["clip"] for row in clips_by_cluster.get(cid, [])]
        sorted_features = sorted(distinctive.items(), key=lambda item: abs(item[1]), reverse=True)
        rows.append(
            {
                "cluster": cid,
                "count": cluster.get("count", 0),
                "auto_name": cluster.get("suggested_name", ""),
                "feature_read": feature_read(distinctive),
                "distinctive_features": "; ".join(f"{k}:{v:+.3f}" for k, v in sorted_features[:6]),
                "representative_clips": " | ".join(clips),
                "human_label": "",
                "notes": "",
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    write_html(rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
