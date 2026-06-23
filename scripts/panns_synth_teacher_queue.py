from __future__ import annotations

import argparse
import csv
import html
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from infer_reference_elements_timeline import load_audio
from panns_source_kind_teacher import load_model, score_batch, source_kind_scores


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_segment(row: dict, out_path: Path, sample_rate: int = 32_000) -> None:
    audio, sr = load_audio(Path(row["stem_path"]), sample_rate=sample_rate)
    start = max(0, int(float(row["start"]) * sr))
    end = min(audio.shape[0], int(float(row["end"]) * sr))
    clip = audio[start:end]
    min_len = int(1.0 * sr)
    if clip.shape[0] < min_len:
        pad = np.zeros((min_len - clip.shape[0], clip.shape[1]), dtype=np.float32)
        clip = np.vstack([clip, pad])
    sf.write(out_path, clip, sr)


def parse_support(value: str) -> set[str]:
    return {part.strip() for part in str(value or "").split("|") if part.strip()}


def format_tags(tags: list[tuple[str, float]]) -> str:
    return "; ".join(f"{label}:{score:.3f}" for label, score in tags)


def format_scores(scores: dict[str, float], threshold: float, limit: int) -> tuple[str, str]:
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    detected = [label for label, score in items if score >= threshold]
    top = "; ".join(
        f"{label}:{score:.3f}{'/detected' if label in detected else '/possible'}"
        for label, score in items[:limit]
    )
    return "|".join(detected[:limit]), top


def agreement(row: dict, detected: set[str]) -> str:
    if not detected:
        return "no_teacher_signal"
    matches = parse_support(row.get("support_matches", ""))
    matched_labels = {part.split(":", 1)[0] for part in matches if part}
    if detected & matched_labels:
        return "supports_current_match"
    if row.get("source_kind_support") in {"strong_support", "soft_support"} and detected:
        return "parallel_evidence"
    if row.get("ensemble_decision") == "needs_review_or_more_data":
        return "teacher_suggests_gap"
    return "teacher_context_only"


def write_html(rows: list[dict], out_html: Path) -> None:
    agreement_counts = Counter(row["panns_agreement"] for row in rows)
    detected_counts = Counter()
    for row in rows:
        detected_counts.update(parse_support(row["panns_detected"]))

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table>"

    detail = []
    for row in rows:
        detail.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td>{html.escape(row['synth_label_top'])} ({row['synth_label_conf']})</td>"
            f"<td>{html.escape(row['ensemble_decision'])}</td>"
            f"<td>{html.escape(row['source_kind_support'])}</td>"
            f"<td>{html.escape(row['panns_detected'] or '-')}</td>"
            f"<td>{html.escape(row['panns_agreement'])}</td>"
            f"<td>{html.escape(row['panns_source_kind_top'])}</td>"
            f"<td>{html.escape(row['panns_raw_top'])}</td>"
            f"<td><audio controls preload='none' src='{html.escape(row['stem_path'])}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>PANNs Synth Teacher Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>PANNs Synth Teacher Queue</h1>
<p class="note">PANNs is a broad AudioSet model. It is useful for source presence checks, not fine synth-design labels.</p>
{count_table("Agreement", agreement_counts)}
{count_table("PANNs Detected Source Kinds", detected_counts)}
<h2>Details</h2>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Specialist</th><th>Decision</th><th>Current Support</th><th>PANNs Detected</th><th>Agreement</th><th>PANNs Mapped Scores</th><th>Raw Tags</th><th>Stem Audio</th></tr>
{''.join(detail)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_queue_v4_strict.csv"))
    parser.add_argument("--cache-root", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_panns_teacher_queue_v4_strict.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_panns_teacher_queue_v4_strict.html"))
    args = parser.parse_args()

    rows = read_rows(args.queue)[: args.limit]
    if not rows:
        raise SystemExit("No queue rows found.")
    model = load_model(args.cache_root)
    out_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip_paths = []
        for idx, row in enumerate(rows):
            clip_path = tmp_dir / f"clip_{idx:05d}.wav"
            write_segment(row, clip_path)
            clip_paths.append(clip_path)
        tag_rows = score_batch(clip_paths, model, args.top_k, args.batch_size)

    for row, tags in zip(rows, tag_rows):
        scores = source_kind_scores(tags)
        detected, mapped_top = format_scores(scores, args.threshold, args.top_k)
        out = dict(row)
        out["panns_detected"] = detected
        out["panns_source_kind_top"] = mapped_top
        out["panns_raw_top"] = format_tags(tags)
        out["panns_agreement"] = agreement(out, parse_support(detected))
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
