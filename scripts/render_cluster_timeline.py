import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


def load_cluster_names(path):
    names = {}
    feature_reads = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cluster = int(row["cluster"])
            names[cluster] = row.get("human_label") or row.get("auto_name") or f"cluster_{cluster}"
            feature_reads[cluster] = row.get("feature_read", "")
    return names, feature_reads


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            row["cluster"] = int(row["cluster"])
            row["segment_index"] = int(row["segment_index"])
            row["start"] = float(row["start"])
            row["end"] = float(row["end"])
            row["cluster_distance"] = float(row.get("cluster_distance", 0.0))
            rows.append(row)
    return rows


def merge_regions(rows):
    by_file = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["file"], r["start"])):
        by_file[row["file"]].append(row)

    regions = []
    for file, items in by_file.items():
        current = None
        for row in items:
            if current and row["cluster"] == current["cluster"] and abs(row["start"] - current["end"]) < 1e-6:
                current["end"] = row["end"]
                current["segments"].append(str(row["segment_index"]))
                current["distances"].append(row["cluster_distance"])
            else:
                if current:
                    regions.append(current)
                current = {
                    "file": file,
                    "start": row["start"],
                    "end": row["end"],
                    "cluster": row["cluster"],
                    "segments": [str(row["segment_index"])],
                    "distances": [row["cluster_distance"]],
                }
        if current:
            regions.append(current)
    return regions


def write_csv(regions, names, feature_reads, path):
    fieldnames = ["file", "start", "end", "cluster", "label", "feature_read", "segments", "mean_cluster_distance"]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for region in regions:
            writer.writerow(
                {
                    "file": region["file"],
                    "start": f"{region['start']:.2f}",
                    "end": f"{region['end']:.2f}",
                    "cluster": region["cluster"],
                    "label": names.get(region["cluster"], f"cluster_{region['cluster']}"),
                    "feature_read": feature_reads.get(region["cluster"], ""),
                    "segments": ",".join(region["segments"]),
                    "mean_cluster_distance": f"{sum(region['distances']) / len(region['distances']):.3f}",
                }
            )


def write_html(regions, names, feature_reads, path):
    style = """
    body{font-family:Arial,sans-serif;margin:24px;color:#111}
    table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:28px}
    th,td{border:1px solid #ddd;padding:8px;vertical-align:top}
    th{background:#eee;text-align:left}
    .label{font-weight:700}
    .distance{color:#666}
    """
    by_file = defaultdict(list)
    for region in regions:
        by_file[region["file"]].append(region)

    sections = []
    for file, items in by_file.items():
        rows = []
        for region in items:
            label = names.get(region["cluster"], f"cluster_{region['cluster']}")
            read = feature_reads.get(region["cluster"], "")
            dist = sum(region["distances"]) / len(region["distances"])
            rows.append(
                "<tr>"
                f"<td>{region['start']:.2f}-{region['end']:.2f}s</td>"
                f"<td>{region['cluster']}</td>"
                f"<td class='label'>{html.escape(label)}</td>"
                f"<td>{html.escape(read)}</td>"
                f"<td>{html.escape(', '.join(region['segments']))}</td>"
                f"<td class='distance'>{dist:.3f}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{html.escape(file)}</h2>"
            "<table><thead><tr><th>Time</th><th>Cluster</th><th>Sound Group</th><th>DSP Read</th><th>Segments</th><th>Distance</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Cluster Sound Timeline</title><style>{style}</style></head>
<body>
<h1>Cluster Sound Timeline</h1>
<p>This view uses unsupervised DSP clusters from the current external clips. Labels are cluster names, not exact instrument or plugin recovery.</p>
{''.join(sections)}
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--label-review", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    names, feature_reads = load_cluster_names(args.label_review)
    rows = load_rows(args.clusters_csv)
    regions = merge_regions(rows)
    write_csv(regions, names, feature_reads, args.out_csv)
    write_html(regions, names, feature_reads, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
