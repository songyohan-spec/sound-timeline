from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_mert_embeddings import SUPPORTED_AUDIO_EXTENSIONS, embed_audio, load_mert


def audio_files(folder: Path, recursive: bool = True) -> list[Path]:
    iterator = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def collect_reference_files(source_root: Path, mapping: dict[str, list[str]], max_files_per_label: int) -> dict[str, list[Path]]:
    by_kind: dict[str, list[Path]] = {}
    for source_kind, training_labels in mapping.items():
        paths: list[Path] = []
        for label in training_labels:
            folder = source_root / label
            if not folder.exists():
                continue
            paths.extend(audio_files(folder, recursive=True)[:max_files_per_label])
        by_kind[source_kind] = paths
    return by_kind


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return (vector / norm).astype(np.float32)
    return vector.astype(np.float32)


def build_centroids(
    by_kind: dict[str, list[Path]],
    processor,
    model,
    torch,
    sample_rate: int,
    device: str,
    max_seconds: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    centroids: dict[str, np.ndarray] = {}
    rows = []
    total = sum(len(paths) for paths in by_kind.values())
    done = 0
    for source_kind, paths in sorted(by_kind.items()):
        vectors = []
        for path in paths:
            done += 1
            print(f"[ref {done}/{total}] {source_kind}: {path}")
            vectors.append(embed_audio(path, processor, model, torch, sample_rate, device, max_seconds))
        if vectors:
            centroids[source_kind] = normalize(np.mean(np.stack(vectors), axis=0))
        rows.append({"source_kind": source_kind, "reference_files": len(paths), "has_centroid": "yes" if vectors else "no"})
    return centroids, rows


def score_targets(
    targets: list[Path],
    centroids: dict[str, np.ndarray],
    processor,
    model,
    torch,
    sample_rate: int,
    device: str,
    max_seconds: float,
    top_k: int,
) -> list[dict]:
    rows = []
    labels = sorted(centroids)
    matrix = np.stack([centroids[label] for label in labels]) if labels else np.zeros((0, 1), dtype=np.float32)
    for index, path in enumerate(targets, 1):
        print(f"[target {index}/{len(targets)}] {path}")
        emb = embed_audio(path, processor, model, torch, sample_rate, device, max_seconds)
        sims = matrix @ emb
        order = np.argsort(sims)[::-1][:top_k]
        rows.append(
            {
                "file": path.as_posix(),
                "top_source_kinds": "; ".join(f"{labels[idx]}:{float(sims[idx]):.4f}" for idx in order),
                "primary_source_kind": labels[int(order[0])] if len(order) else "",
                "primary_similarity": round(float(sims[int(order[0])]), 6) if len(order) else 0.0,
            }
        )
    return rows


def write_html(ref_rows: list[dict], target_rows: list[dict], out_html: Path) -> None:
    ref_body = "".join(
        f"<tr><td>{html.escape(row['source_kind'])}</td><td>{row['reference_files']}</td><td>{row['has_centroid']}</td></tr>"
        for row in ref_rows
    )
    target_body = "".join(
        "<tr>"
        f"<td>{html.escape(Path(row['file']).name)}</td>"
        f"<td><audio controls preload='metadata' src='{html.escape(Path(row['file']).as_posix())}'></audio></td>"
        f"<td>{html.escape(row['primary_source_kind'])}</td>"
        f"<td>{row['primary_similarity']}</td>"
        f"<td>{html.escape(row['top_source_kinds'])}</td>"
        "</tr>"
        for row in target_rows
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>MERT Source-Kind Centroid Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 190px; }}
</style>
<h1>MERT Source-Kind Centroid Report</h1>
<p>MERT is used as an open-source music embedding model. This report compares target clips/stems against source-kind centroids built from the local reference training folders. Similarity is not ground truth, but it is a stronger music-aware check than DSP-only features.</p>
<h2>Reference Centroids</h2>
<table><tr><th>Source Kind</th><th>Reference Files</th><th>Centroid</th></tr>{ref_body}</table>
<h2>Target Similarity</h2>
<table><tr><th>Target</th><th>Audio</th><th>Primary Source Kind</th><th>Similarity</th><th>Top Source Kinds</th></tr>{target_body}</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--map", type=Path, default=Path("configs/source_kind_training_map.json"))
    parser.add_argument("--target-dir", type=Path, default=Path("outputs/demucs_stems_test/htdemucs"))
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--max-files-per-label", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_test/mert_source_kind_centroid.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_test/mert_source_kind_centroid.html"))
    args = parser.parse_args()

    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    targets = audio_files(args.target_dir, recursive=True)
    if not targets:
        raise SystemExit(f"No target audio files found in {args.target_dir}")

    processor, model, torch = load_mert(args.model_name, args.device)
    by_kind = collect_reference_files(args.source_root, mapping, args.max_files_per_label)
    centroids, ref_rows = build_centroids(by_kind, processor, model, torch, args.sample_rate, args.device, args.max_seconds)
    if not centroids:
        raise SystemExit("No centroids were built. Check source-root and mapping.")
    target_rows = score_targets(targets, centroids, processor, model, torch, args.sample_rate, args.device, args.max_seconds, args.top_k)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(target_rows[0].keys()))
        writer.writeheader()
        writer.writerows(target_rows)
    write_html(ref_rows, target_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
