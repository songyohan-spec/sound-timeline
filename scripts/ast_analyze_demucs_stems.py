from __future__ import annotations

import argparse
import csv
import html
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_demucs_stems_source_kind import allowed_for_stem, discover_stems
from ast_source_kind_teacher import ast_source_kind_scores
from audioset_teacher_review_queue import score_audio
from infer_reference_elements_timeline import load_audio


def parse_top(tags: list[dict]) -> str:
    return "; ".join(f"{tag['label']}:{tag['score']:.3f}" for tag in tags)


def load_ast_model(model_name: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

    try:
        extractor = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForAudioClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        extractor = AutoFeatureExtractor.from_pretrained(model_name)
        model = AutoModelForAudioClassification.from_pretrained(model_name)
    model.eval()
    return extractor, model, torch


def detected_source_kinds(tags: list[dict], stem: str, threshold: float, top_k: int) -> tuple[str, str]:
    pairs = [(tag["label"], float(tag["score"])) for tag in tags]
    scores = ast_source_kind_scores(pairs)
    ordered = [
        (label, score)
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if allowed_for_stem(stem, label)
    ]
    detected = [label for label, score in ordered if score >= threshold]
    top = "; ".join(
        f"{label}:{score:.3f}{'/detected' if label in detected else '/possible'}"
        for label, score in ordered[:top_k]
    )
    return "|".join(detected[:top_k]), top


def existing_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return {
            (row["track"], row["stem"], row["start"], row["end"])
            for row in csv.DictReader(file)
        }


def write_html(csv_path: Path, out_html: Path, stems: list[str] | None = None) -> None:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    by_stem: dict[str, Counter[str]] = defaultdict(Counter)
    all_counts = Counter()
    for row in rows:
        labels = [label for label in row["ast_source_kinds"].split("|") if label]
        by_stem[row["stem"]].update(labels)
        all_counts.update(labels)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    detail = []
    for row in rows:
        stem_path = Path(row["stem_path"])
        try:
            audio_rel = stem_path.relative_to(out_html.parent).as_posix()
        except ValueError:
            audio_rel = stem_path.as_posix()
        detail.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload='metadata' src='{html.escape(audio_rel)}'></audio></td>"
            f"<td>{html.escape(row['ast_source_kinds'] or '-')}</td>"
            f"<td>{html.escape(row['ast_source_kind_top'])}</td>"
            f"<td>{html.escape(row['audioset_top'])}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AST Demucs Stem Source-Kind Teacher</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
</style>
<h1>AST Demucs Stem Source-Kind Teacher</h1>
<p>Open-source AST/AudioSet tags are run on each Demucs stem segment, then mapped into the project source-kind vocabulary with stem-aware filtering.</p>
{count_table("All AST Source Kinds", all_counts)}
{''.join(count_table(f"{stem} stem", by_stem[stem]) for stem in (stems or sorted(by_stem)))}
<h2>Segment Detail</h2>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Stem Audio</th><th>AST Source Kinds</th><th>Mapped Scores</th><th>Raw AST Tags</th></tr>
{''.join(detail)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_full/htdemucs"))
    parser.add_argument("--model-name", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--stems", default="", help="Comma-separated stems to analyze. Defaults to discovered Demucs stems.")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_full/ast_stem_source_kind.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_full/ast_stem_source_kind.html"))
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    seen = existing_keys(args.out_csv)
    write_header = not args.out_csv.exists()
    extractor, model, torch = load_ast_model(args.model_name)
    stems = discover_stems(args.stems_root, args.stems)

    fieldnames = ["track", "stem", "start", "end", "ast_source_kinds", "ast_source_kind_top", "audioset_top", "stem_path"]
    processed = 0
    with args.out_csv.open("a", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for track_dir in sorted(path for path in args.stems_root.iterdir() if path.is_dir()):
                for stem in stems:
                    stem_path = track_dir / f"{stem}.wav"
                    if not stem_path.exists():
                        continue
                    audio, sr = load_audio(stem_path)
                    segment_len = max(1, int(args.segment_seconds * sr))
                    hop_len = max(1, int(args.hop_seconds * sr))
                    starts = list(range(0, max(1, len(audio) - segment_len + 1), hop_len)) or [0]
                    for index, start in enumerate(starts):
                        end = min(len(audio), start + segment_len)
                        start_s = str(round(start / sr, 4))
                        end_s = str(round(min(len(audio) / sr, (start + segment_len) / sr), 4))
                        key = (track_dir.name, stem, start_s, end_s)
                        if key in seen:
                            continue
                        clip = audio[start:end]
                        if len(clip) < segment_len:
                            pad = np.zeros((segment_len - len(clip), clip.shape[1]), dtype=np.float32)
                            clip = np.vstack([clip, pad])
                        clip_path = tmp_dir / f"{track_dir.name}_{stem}_{index:04d}.wav"
                        sf.write(clip_path, clip, sr)
                        print(f"[{processed + 1}] {track_dir.name} {stem} {start_s}-{end_s}s")
                        tags = score_audio(clip_path, extractor, model, torch, args.top_k)
                        detected, mapped_top = detected_source_kinds(tags, stem, args.threshold, args.top_k)
                        writer.writerow(
                            {
                                "track": track_dir.name,
                                "stem": stem,
                                "start": start_s,
                                "end": end_s,
                                "ast_source_kinds": detected,
                                "ast_source_kind_top": mapped_top,
                                "audioset_top": parse_top(tags),
                                "stem_path": str(stem_path),
                            }
                        )
                        file.flush()
                        processed += 1
                        if args.limit and processed >= args.limit:
                            write_html(args.out_csv, args.out_html, stems)
                            print(f"wrote: {args.out_csv}")
                            print(f"wrote: {args.out_html}")
                            return

    write_html(args.out_csv, args.out_html, stems)
    print(f"processed new rows: {processed}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
