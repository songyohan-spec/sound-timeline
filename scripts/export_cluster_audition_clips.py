from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import soundfile as sf


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", type=Path, default=Path("outputs/dsp_segment_clusters_v4_hq_stereo.csv"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cluster_audition_clips"))
    parser.add_argument("--per-cluster", type=int, default=3)
    args = parser.parse_args()

    with args.clusters.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cluster[row["cluster"]].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cluster, cluster_rows in sorted(by_cluster.items(), key=lambda item: int(item[0])):
        selected = sorted(cluster_rows, key=lambda row: float(row.get("cluster_distance", 999.0)))[: args.per_cluster]
        cluster_dir = args.out_dir / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        for rank, row in enumerate(selected, start=1):
            audio_path = args.audio_dir / row["file"]
            audio, sr = sf.read(audio_path, always_2d=True)
            start = float(row["start"])
            end = float(row["end"])
            start_i = max(0, int(start * sr))
            end_i = min(len(audio), int(end * sr))
            clip = audio[start_i:end_i]
            out_name = f"{rank:02d}_{safe_name(row['stem'])}_seg{int(float(row['segment_index'])):03d}_{start:.2f}-{end:.2f}s.wav"
            out_path = cluster_dir / out_name
            sf.write(out_path, clip, sr)
            manifest.append(
                {
                    "cluster": cluster,
                    "rank": rank,
                    "source_file": row["file"],
                    "segment_index": row["segment_index"],
                    "start": row["start"],
                    "end": row["end"],
                    "clip": str(out_path),
                    "cluster_distance": row.get("cluster_distance", ""),
                }
            )

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"wrote: {manifest_path}")
    print(f"clips: {len(manifest)}")


if __name__ == "__main__":
    main()
