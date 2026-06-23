from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(path, always_2d=True)
        return audio.astype(np.float32), sr
    except Exception:
        import librosa

        loaded, sr = librosa.load(path, sr=None, mono=False)
        if loaded.ndim == 1:
            audio = np.stack([loaded, loaded], axis=1)
        else:
            audio = loaded.T
        return audio.astype(np.float32), sr


def parse_candidate(value: str) -> tuple[str, float, float] | None:
    if ":" not in value or "/" not in value:
        return None
    label, rest = value.rsplit(":", 1)
    score, threshold = rest.split("/", 1)
    try:
        return label.strip(), float(score), float(threshold)
    except ValueError:
        return None


def top_candidates(row: dict, limit: int = 5) -> list[tuple[str, float, float]]:
    out = []
    for part in str(row.get("top_labels", "")).split(";"):
        parsed = parse_candidate(part.strip())
        if parsed:
            out.append(parsed)
    return out[:limit]


def priority_score(row: dict, mode: str = "detected") -> float:
    if mode == "all":
        return 0.0
    detected = [x for x in str(row.get("detected_labels", "")).split("|") if x]
    candidates = top_candidates(row, 5)
    if not candidates:
        return 0.0
    top_label, top_score, threshold = candidates[0]
    margin = top_score - threshold
    if mode == "ambiguous":
        # Highest priority goes to cases just below or just above threshold.
        return max(0.0, 1.0 - abs(margin) * 6.0) + (0.15 if not detected else 0.0)
    # Prioritize confident detections and near-threshold ambiguous cases.
    if detected:
        return 2.0 + max(0.0, margin)
    if -0.10 <= margin < 0.0:
        return 1.2 + (0.10 + margin)
    if top_score >= 0.30:
        return 1.0
    return top_score


def crop_segment(audio_root: Path, row: dict, out_dir: Path, index: int, pad: float) -> Path:
    src = audio_root / row["file"]
    audio, sr = load_audio(src)
    start = max(0.0, float(row["start"]) - pad)
    end = min(len(audio) / sr, float(row["end"]) + pad)
    start_i = max(0, int(start * sr))
    end_i = min(len(audio), int(end * sr))
    clip = audio[start_i:end_i]
    safe_stem = Path(row["file"]).stem.replace(" ", "_")
    out_file = out_dir / f"review_{index:04d}_{safe_stem}_{float(row['start']):.2f}_{float(row['end']):.2f}.wav"
    sf.write(out_file, clip, sr)
    return out_file


def write_html(rows: list[dict], path: Path) -> None:
    body = []
    for idx, row in enumerate(rows):
        clip_src = Path(row["clip"]).as_posix()
        feedback_line = (
            f"{row['file']},{row['start']},{row['end']},"
            "GROUPS_TO_FORBID,LABELS_TO_FORBID,FORCE_LABELS,NOTE"
        )
        body.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(clip_src)}\"></audio></td>"
            f"<td>{html.escape(row.get('detected_groups', '') or '-')}</td>"
            f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
            f"<td>{html.escape(row.get('top_labels', ''))}</td>"
            f"<td><code>{html.escape(feedback_line)}</code></td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Segment Review Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 240px; }}
code {{ white-space: pre-wrap; font-size: 12px; }}
.note {{ color: #444; }}
</style>
<h1>Segment Review Queue</h1>
<p class="note">Listen to clips and copy a feedback line when a detection is wrong. Replace placeholders before adding to configs/segment_feedback.csv.</p>
<table>
<tr><th>#</th><th>Segment</th><th>Audio</th><th>Detected Groups</th><th>Detected Labels</th><th>Top Candidates</th><th>Feedback Template</th></tr>
{''.join(body)}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/external_reference_ensemble_batch_suppressed.csv"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/review_queue"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--pad", type=float, default=0.25)
    parser.add_argument("--mode", choices=["detected", "ambiguous", "all"], default="detected")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clip_dir = args.out_dir / "audio"
    clip_dir.mkdir(exist_ok=True)

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.mode != "all":
        rows.sort(key=lambda row: priority_score(row, args.mode), reverse=True)
    selected = rows[: args.limit]
    out_rows = []
    for idx, row in enumerate(selected):
        clip = crop_segment(args.audio_root, row, clip_dir, idx, args.pad)
        new_row = dict(row)
        new_row["clip"] = clip.relative_to(args.out_dir).as_posix()
        new_row["priority_score"] = round(priority_score(row, args.mode), 5)
        out_rows.append(new_row)

    csv_path = args.out_dir / "review_queue.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    html_path = args.out_dir / "review_queue.html"
    write_html(out_rows, html_path)
    print(f"rows: {len(out_rows)}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {html_path}")


if __name__ == "__main__":
    main()
