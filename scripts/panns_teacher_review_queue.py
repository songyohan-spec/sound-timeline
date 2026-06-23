from __future__ import annotations

import argparse
import csv
import html
import os
import urllib.request
from pathlib import Path

import librosa
import numpy as np


LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def prepare_panns_home(cache_root: Path) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    panns_dir = cache_root / "panns_data"
    panns_dir.mkdir(parents=True, exist_ok=True)
    labels = panns_dir / "class_labels_indices.csv"
    checkpoint = panns_dir / "Cnn14_mAP=0.431.pth"
    if not labels.exists():
        print(f"downloading labels: {labels}")
        urllib.request.urlretrieve(LABELS_URL, labels)
    if not checkpoint.exists() or checkpoint.stat().st_size < 300_000_000:
        print(f"downloading PANNs checkpoint: {checkpoint}")
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    # panns-inference uses Path.home()/panns_data at import time.
    os.environ["USERPROFILE"] = str(cache_root.resolve())
    os.environ["HOME"] = str(cache_root.resolve())
    return checkpoint


def load_audio(path: Path, target_sr: int = 32_000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def load_model(cache_root: Path):
    checkpoint = prepare_panns_home(cache_root)
    from panns_inference import AudioTagging

    return AudioTagging(checkpoint_path=str(checkpoint), device="cpu")


def top_tags(scores: np.ndarray, labels: list[str], top_k: int) -> list[dict]:
    order = np.argsort(scores)[::-1][:top_k]
    return [
        {"label": str(labels[int(idx)]), "score": round(float(scores[int(idx)]), 5)}
        for idx in order
    ]


def score_batch(paths: list[Path], model, top_k: int, batch_size: int) -> list[list[dict]]:
    out: list[list[dict]] = []
    for batch_start in range(0, len(paths), batch_size):
        batch_paths = paths[batch_start : batch_start + batch_size]
        audios = [load_audio(path) for path in batch_paths]
        max_len = max(len(audio) for audio in audios)
        batch = np.zeros((len(audios), max_len), dtype=np.float32)
        for idx, audio in enumerate(audios):
            batch[idx, : len(audio)] = audio
        clipwise, _ = model.inference(batch)
        for scores in clipwise:
            out.append(top_tags(scores, model.labels, top_k))
    return out


def broad_support(tags: list[dict]) -> set[str]:
    text = " | ".join(tag["label"].lower() for tag in tags)
    support = set()
    if any(word in text for word in ["singing", "vocal", "choir", "speech", "voice", "rapping"]):
        support.add("vocals")
    if any(word in text for word in ["drum", "snare", "cymbal", "hi-hat", "percussion", "beat", "rimshot"]):
        support.add("drums")
    if any(word in text for word in ["bass", "sub-bass"]):
        support.add("bass")
    if any(word in text for word in ["guitar", "strum"]):
        support.add("guitar_strings")
    if any(word in text for word in ["synth", "electronic music", "keyboard", "sampler"]):
        support.add("synth")
    if any(word in text for word in ["noise", "static", "click", "glitch", "effects", "whoosh"]):
        support.add("noise_fx")
    return support


def split_pipe(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").split("|") if item.strip()}


def write_html(rows: list[dict], path: Path) -> None:
    body = []
    for idx, row in enumerate(rows):
        body.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row['clip'])}\"></audio></td>"
            f"<td>{html.escape(row.get('detected_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
            f"<td>{html.escape(row.get('panns_groups', '') or '-')}</td>"
            f"<td>{html.escape(row['panns_top'])}</td>"
            f"<td>{html.escape(row['panns_support'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>PANNs Teacher Review Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 240px; }}
.note {{ color: #444; }}
</style>
<h1>PANNs Teacher Review Queue</h1>
<p class="note">PANNs is an AudioSet-trained public model. It gives broad audio-tag evidence, not exact sound-design labels.</p>
<table>
<tr><th>#</th><th>Segment</th><th>Audio</th><th>Our Groups</th><th>Our Labels</th><th>PANNs Groups</th><th>PANNs Top Tags</th><th>Support</th></tr>
{''.join(body)}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/review_queue_detected/review_queue.csv"))
    parser.add_argument("--cache-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/review_queue_detected/panns_teacher.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/review_queue_detected/panns_teacher.html"))
    args = parser.parse_args()

    model = load_model(args.cache_root)
    with args.queue.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    clip_paths = [args.queue.parent / row["clip"] for row in rows]
    for idx, clip_path in enumerate(clip_paths, 1):
        print(f"[{idx}/{len(rows)}] {clip_path}")
    batch_tags = score_batch(clip_paths, model, args.top_k, args.batch_size)

    out_rows = []
    for row, tags in zip(rows, batch_tags):
        support = broad_support(tags)
        our_groups = split_pipe(row.get("detected_groups", ""))
        if support & our_groups:
            relation = "support"
        elif support:
            relation = "disagree_or_missing"
        else:
            relation = "no_clear_teacher_group"
        out = dict(row)
        out["panns_top"] = "; ".join(f"{tag['label']}:{tag['score']:.3f}" for tag in tags)
        out["panns_groups"] = "|".join(sorted(support))
        out["panns_support"] = relation
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    write_html(out_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
