import argparse
import csv
import json
from pathlib import Path

import numpy as np


METADATA_COLUMNS = {"file", "stem", "segment_index", "start", "end", "cluster", "cluster_distance", "pca_x", "pca_y"}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_labels(path):
    labels = {}
    notes = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster"])
            labels[cid] = row.get("human_label") or row.get("auto_name") or f"cluster_{cid}"
            notes[cid] = row.get("notes", "")
    return labels, notes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--label-review", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_csv(args.clusters_csv)
    labels, notes = load_labels(args.label_review)
    feature_cols = [col for col in rows[0].keys() if col not in METADATA_COLUMNS]
    x = np.array([[float(row[col]) for col in feature_cols] for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-8
    z = (x - mean) / std

    centroids = {}
    counts = {}
    for cid in sorted({int(row["cluster"]) for row in rows}):
        idx = [i for i, row in enumerate(rows) if int(row["cluster"]) == cid]
        centroids[str(cid)] = z[idx].mean(axis=0).tolist()
        counts[str(cid)] = len(idx)

    model = {
        "model_type": "dsp_palette_centroid",
        "feature_columns": feature_cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "clusters": {
            str(cid): {
                "label": labels.get(cid, f"cluster_{cid}"),
                "notes": notes.get(cid, ""),
                "count": counts[str(cid)],
                "centroid": centroids[str(cid)],
            }
            for cid in sorted(labels)
            if str(cid) in centroids
        },
        "caution": "This model maps audio segments to learned external-clip sound groups, not exact instruments, stems, plugins, or presets.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
