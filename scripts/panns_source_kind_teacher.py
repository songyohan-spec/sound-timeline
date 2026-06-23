from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter
from pathlib import Path

import librosa
import numpy as np


AUDIOSET_LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def prepare_panns_home(cache_root: Path) -> None:
    import urllib.request

    cache_root.mkdir(parents=True, exist_ok=True)
    panns_dir = cache_root / "panns_data"
    panns_dir.mkdir(parents=True, exist_ok=True)
    labels = panns_dir / "class_labels_indices.csv"
    checkpoint = panns_dir / "Cnn14_mAP=0.431.pth"
    if not labels.exists():
        urllib.request.urlretrieve(AUDIOSET_LABELS_URL, labels)
    if not checkpoint.exists() or checkpoint.stat().st_size < 300_000_000:
        urllib.request.urlretrieve(PANNS_CHECKPOINT_URL, checkpoint)
    os.environ["USERPROFILE"] = str(cache_root.resolve())
    os.environ["HOME"] = str(cache_root.resolve())


def load_model(cache_root: Path):
    prepare_panns_home(cache_root)
    from panns_inference import AudioTagging

    return AudioTagging(checkpoint_path=str(cache_root / "panns_data" / "Cnn14_mAP=0.431.pth"), device="cpu")


def load_audio(path: Path, target_sr: int = 32_000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32)


def top_tags(scores: np.ndarray, labels: list[str], top_k: int) -> list[tuple[str, float]]:
    order = np.argsort(scores)[::-1][:top_k]
    return [(str(labels[int(idx)]), float(scores[int(idx)])) for idx in order]


def score_batch(paths: list[Path], model, top_k: int, batch_size: int) -> list[list[tuple[str, float]]]:
    outputs: list[list[tuple[str, float]]] = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        audios = [load_audio(path) for path in chunk]
        max_len = max(len(audio) for audio in audios)
        batch = np.zeros((len(audios), max_len), dtype=np.float32)
        for idx, audio in enumerate(audios):
            batch[idx, : len(audio)] = audio
        clipwise, _ = model.inference(batch)
        for scores in clipwise:
            outputs.append(top_tags(scores, model.labels, top_k))
        print(f"processed {min(start + batch_size, len(paths))}/{len(paths)}")
    return outputs


def source_kind_scores(tags: list[tuple[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = {}

    def add(label: str, value: float) -> None:
        scores[label] = max(scores.get(label, 0.0), value)

    for tag, score in tags:
        lower = tag.lower()
        if any(word in lower for word in ["singing", "vocal music", "female singing", "male singing"]):
            add("clean_or_lead_vocal", score)
            add("lead_or_hook_vocal", score * 0.9)
        if any(word in lower for word in ["speech", "rapping", "narration"]):
            add("rap_or_spoken_vocal", score)
        if any(word in lower for word in ["choir", "chorus"]):
            add("vocal_pad_or_harmony", score)
        if any(word in lower for word in ["synthesizer", "keyboard", "electronic music", "electronica"]):
            add("synth_pad_or_wash", score * 0.55)
            add("digital_synth_lead", score * 0.35)
        if any(word in lower for word in ["ringtone", "ding", "jingle", "bell", "tinkle", "chink", "clink"]):
            add("synth_pluck_or_bell", score)
        if any(word in lower for word in ["sampler", "sample"]):
            add("sampled_loop_texture", score)
        if any(word in lower for word in ["bass", "sub-bass", "dubstep", "drum and bass"]):
            add("sub_or_808_bass", score * 0.65)
            add("synth_bass", score * 0.55)
        if any(word in lower for word in ["drum machine", "beat"]):
            add("electronic_drum_machine", score)
        if any(word in lower for word in ["drum kit", "breakbeat", "drum roll"]):
            add("breakbeat_or_live_drums", score)
        if any(word in lower for word in ["bass drum", "kick drum", "thump"]):
            add("kick_or_low_hit", score)
        if any(word in lower for word in ["snare", "clap", "snap"]):
            add("snare_clap_or_snap", score)
        if any(word in lower for word in ["hi-hat", "cymbal", "tick", "click"]):
            add("hat_tick_or_click", score)
        if any(word in lower for word in ["click", "mechanical", "glitch"]):
            add("glitch_percussion", score * 0.75)
        if any(word in lower for word in ["guitar", "strum", "plucked string"]):
            add("guitar_or_plucked_loop", score)
        if any(word in lower for word in ["electric guitar", "rock", "grunge", "metal"]):
            add("distorted_guitar_or_rock_texture", score * 0.70)
        if any(word in lower for word in ["violin", "string", "cello"]):
            add("string_or_violin_like", score)
        if any(word in lower for word in ["piano", "electric piano", "organ"]):
            add("piano_or_keyboard_loop", score)
            add("warm_keys_or_organ", score * 0.85)
        if any(word in lower for word in ["noise", "static", "whoosh", "sound effect", "burst", "explosion"]):
            add("noise_or_fx_transition", score)
        if any(word in lower for word in ["whoosh", "wind"]):
            add("riser_or_swell", score)
        if any(word in lower for word in ["bang", "burst", "explosion", "thump"]):
            add("impact_or_tail", score)
    return scores


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def load_heuristic(path: Path) -> dict[tuple[str, str, str], dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows[(row["file"], str(row["start"]), str(row["end"]))] = row
    return rows


def format_scores(scores: dict[str, float], threshold: float, limit: int) -> str:
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return "; ".join(f"{label}:{score:.3f}{'/detected' if score >= threshold else '/possible'}" for label, score in items[:limit])


def write_html(rows: list[dict], out_html: Path) -> None:
    relation_counts = Counter(row["agreement"] for row in rows)
    detected_counts = Counter()
    for row in rows:
        detected_counts.update(split_pipe(row["panns_detected"]))

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    details = []
    for row in rows:
        details.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
            f"<td>{html.escape(row['heuristic_active'] or '-')}</td>"
            f"<td>{html.escape(row['panns_detected'] or '-')}</td>"
            f"<td>{html.escape(row['overlap'] or '-')}</td>"
            f"<td>{html.escape(row['agreement'])}</td>"
            f"<td>{html.escape(row['panns_source_kind_top'])}</td>"
            f"<td>{html.escape(row['panns_raw_top'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>PANNs Source Kind Teacher</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
</style>
<h1>PANNs Source Kind Teacher</h1>
<p>PANNs is an open-source AudioSet-trained model. This report maps its public audio tags into our source-kind vocabulary and compares them with the current source-kind report.</p>
{count_table("Agreement", relation_counts)}
{count_table("PANNs Detected Source Kinds", detected_counts)}
<h2>Segment Detail</h2>
<table>
<tr><th>Segment</th><th>Audio</th><th>Current Source Kinds</th><th>PANNs Source Kinds</th><th>Overlap</th><th>Agreement</th><th>PANNs Mapped Scores</th><th>PANNs Raw Tags</th></tr>
{''.join(details)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/review_queue.csv"))
    parser.add_argument("--queue-root", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue"))
    parser.add_argument("--heuristic", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.csv"))
    parser.add_argument("--cache-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/panns_source_kind_teacher.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/panns_source_kind_teacher.html"))
    args = parser.parse_args()

    model = load_model(args.cache_root)
    heuristic = load_heuristic(args.heuristic)
    with args.queue.open("r", encoding="utf-8-sig", newline="") as file:
        queue_rows = list(csv.DictReader(file))[: args.limit]

    clip_paths = []
    for row in queue_rows:
        clip = Path(row["clip"])
        clip_paths.append(clip if clip.is_absolute() else args.queue_root / clip)
    tag_batches = score_batch(clip_paths, model, args.top_k, args.batch_size)

    out_rows = []
    for row, tags in zip(queue_rows, tag_batches):
        scores = source_kind_scores(tags)
        detected = [label for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score >= args.threshold]
        key = (row["file"], str(row["start"]), str(row["end"]))
        heuristic_active = split_pipe(heuristic.get(key, {}).get("active_source_kinds", ""))
        overlap = sorted(set(detected) & set(heuristic_active))
        if overlap:
            agreement = "overlap"
        elif detected and heuristic_active:
            agreement = "disagree"
        elif detected:
            agreement = "panns_only"
        elif heuristic_active:
            agreement = "heuristic_only"
        else:
            agreement = "empty"
        out = dict(row)
        out["heuristic_active"] = "|".join(heuristic_active)
        out["panns_detected"] = "|".join(detected)
        out["overlap"] = "|".join(overlap)
        out["agreement"] = agreement
        out["panns_source_kind_top"] = format_scores(scores, args.threshold, args.top_k)
        out["panns_raw_top"] = "; ".join(f"{label}:{score:.3f}" for label, score in tags)
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    write_html(out_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
