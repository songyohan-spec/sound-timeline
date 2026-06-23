from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METADATA_COLUMNS = {"file", "stem", "segment_index", "start", "end"}


def standardize(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mean) / std


def kmeans_numpy(x, clusters: int, iterations: int = 80, seed: int = 41):
    rng = np.random.default_rng(seed)
    if len(x) < clusters:
        raise SystemExit(f"Need at least {clusters} rows for clustering.")
    centers = x[rng.choice(len(x), size=clusters, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(clusters):
            members = x[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)
            else:
                centers[cluster_id] = x[rng.integers(0, len(x))]
    return labels, centers


def cluster_name_from_z(z_means: pd.Series) -> str:
    high = set(z_means[z_means > 0.45].index)
    low = set(z_means[z_means < -0.45].index)
    parts = []
    if "centroid" in high or "rolloff" in high:
        parts.append("bright")
    if "centroid" in low and "rolloff" in low:
        parts.append("dark")
    if "flatness" in high or "zcr" in high:
        parts.append("noisy")
    if "width" in high:
        parts.append("wide")
    if "width" in low:
        parts.append("narrow")
    if "motion_strength" in high or "rms_std" in high or "rms_range" in high:
        parts.append("dynamic")
    if "motion_strength" in low and "rms_std" in low:
        parts.append("steady")
    if not parts:
        parts.append("mid")
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/dsp_segment_features.csv"))
    parser.add_argument("--clusters", type=int, default=6)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/dsp_segment_clusters.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/dsp_segment_clusters.html"))
    args = parser.parse_args()

    frame = pd.read_csv(args.features, encoding="utf-8-sig")
    feature_cols = [col for col in frame.columns if col not in METADATA_COLUMNS]
    x = frame[feature_cols].astype(float).to_numpy()
    scaled = standardize(x)
    labels, centers = kmeans_numpy(scaled, args.clusters)
    x_axis = feature_cols.index("centroid") if "centroid" in feature_cols else 0
    y_axis = feature_cols.index("motion_strength") if "motion_strength" in feature_cols else min(1, len(feature_cols) - 1)
    coords = scaled[:, [x_axis, y_axis]]

    out = frame.copy()
    out["cluster"] = labels
    distances = np.sqrt(np.sum((scaled - centers[labels]) ** 2, axis=1))
    out["cluster_distance"] = distances
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    cluster_summary = []
    for cluster_id in sorted(out["cluster"].unique()):
        group = out[out["cluster"] == cluster_id]
        means = group[feature_cols].mean().sort_values(ascending=False)
        z_means = pd.Series(scaled[out["cluster"].to_numpy() == cluster_id].mean(axis=0), index=feature_cols).sort_values(ascending=False)
        representatives = (
            group.sort_values("cluster_distance")
            [["file", "stem", "segment_index", "start", "end", "cluster_distance"]]
            .head(5)
            .to_dict(orient="records")
        )
        cluster_summary.append(
            {
                "cluster": int(cluster_id),
                "suggested_name": cluster_name_from_z(z_means),
                "count": int(len(group)),
                "top_files": group["stem"].value_counts().head(5).to_dict(),
                "mean_features": {key: round(float(value), 4) for key, value in means.items()},
                "distinctive_features": {key: round(float(value), 3) for key, value in z_means.head(6).items()},
                "low_features": {key: round(float(value), 3) for key, value in z_means.tail(4).items()},
                "representatives": representatives,
            }
        )

    rows = []
    for item in cluster_summary:
        distinctive = "<br>".join(
            f"{key}: {value}"
            for key, value in item["distinctive_features"].items()
        )
        low_features = "<br>".join(
            f"{key}: {value}"
            for key, value in item["low_features"].items()
        )
        raw_features = "<br>".join(
            f"{key}: {value}"
            for key, value in list(item["mean_features"].items())[:6]
        )
        top_files = "<br>".join(f"{key}: {value}" for key, value in item["top_files"].items())
        reps = "<br>".join(
            f"{row['file']} [{row['start']}-{row['end']}s]"
            for row in item["representatives"]
        )
        rows.append(
            "<tr>"
            f"<td>{item['cluster']}</td>"
            f"<td>{item['suggested_name']}</td>"
            f"<td>{item['count']}</td>"
            f"<td>{distinctive}</td>"
            f"<td>{low_features}</td>"
            f"<td>{raw_features}</td>"
            f"<td>{top_files}</td>"
            f"<td>{reps}</td>"
            "</tr>"
        )

    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>DSP Segment Clusters</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}</style>",
            "<h1>DSP Segment Clusters</h1>",
            f"<p>Segments: <strong>{len(out)}</strong> / Clusters: <strong>{args.clusters}</strong></p>",
            "<table><thead><tr><th>Cluster</th><th>Suggested Name</th><th>Count</th><th>Distinctive Features (z)</th><th>Low Features (z)</th><th>Raw High Means</th><th>Top Files</th><th>Representative Segments</th></tr></thead><tbody>",
            *rows,
            "</tbody></table>",
        ]
    )
    args.out_html.write_text(html, encoding="utf-8")

    summary_json = args.out_html.with_suffix(".json")
    summary_json.write_text(json.dumps(cluster_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")
    print(f"wrote: {summary_json}")


if __name__ == "__main__":
    main()
