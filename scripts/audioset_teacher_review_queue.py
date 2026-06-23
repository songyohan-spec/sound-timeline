from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
    except Exception:
        import librosa

        mono, sr = librosa.load(path, sr=None, mono=True)
        mono = mono.astype(np.float32)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return mono, sr


def load_model(model_name: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    return extractor, model, torch


def score_audio(path: Path, extractor, model, torch, top_k: int) -> list[dict]:
    audio, sr = load_audio(path)
    inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0]
        probs = logits.sigmoid()
        values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
    labels = model.config.id2label
    return [
        {"label": str(labels[int(idx)]), "score": round(float(score), 5)}
        for score, idx in zip(values.cpu(), indices.cpu())
    ]


def broad_support(tags: list[dict]) -> set[str]:
    text = " | ".join(tag["label"].lower() for tag in tags)
    support = set()
    if any(word in text for word in ["singing", "vocal", "choir", "speech", "voice"]):
        support.add("vocals")
    if any(word in text for word in ["drum", "snare", "cymbal", "hi-hat", "percussion", "beat"]):
        support.add("drums")
    if any(word in text for word in ["bass", "sub-bass"]):
        support.add("bass")
    if any(word in text for word in ["guitar", "strum"]):
        support.add("guitar_strings")
    if any(word in text for word in ["synth", "electronic music", "keyboard", "sampler"]):
        support.add("synth")
    if any(word in text for word in ["noise", "static", "click", "glitch", "effects"]):
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
            f"<td>{html.escape(row['audioset_top'])}</td>"
            f"<td>{html.escape(row['teacher_support'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AudioSet Teacher Review Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 240px; }}
.note {{ color: #444; }}
</style>
<h1>AudioSet Teacher Review Queue</h1>
<p class="note">AudioSet/AST tags are broad public-model hints. They can support or challenge our labels, but they are not sound-design ground truth.</p>
<table>
<tr><th>#</th><th>Segment</th><th>Audio</th><th>Our Groups</th><th>Our Labels</th><th>AudioSet Top Tags</th><th>Teacher Support</th></tr>
{''.join(body)}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/review_queue_detected/review_queue.csv"))
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/review_queue_detected/audioset_teacher.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/review_queue_detected/audioset_teacher.html"))
    args = parser.parse_args()

    extractor, model, torch = load_model(args.model_name)
    with args.queue.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    out_rows = []
    for idx, row in enumerate(rows, 1):
        clip_path = args.queue.parent / row["clip"]
        print(f"[{idx}/{len(rows)}] {clip_path}")
        tags = score_audio(clip_path, extractor, model, torch, args.top_k)
        support = broad_support(tags)
        our_groups = split_pipe(row.get("detected_groups", ""))
        if support & our_groups:
            relation = "support"
        elif support:
            relation = "disagree_or_missing"
        else:
            relation = "no_clear_teacher_group"
        out = dict(row)
        out["audioset_top"] = "; ".join(f"{tag['label']}:{tag['score']:.3f}" for tag in tags)
        out["audioset_groups"] = "|".join(sorted(support))
        out["teacher_support"] = relation
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
