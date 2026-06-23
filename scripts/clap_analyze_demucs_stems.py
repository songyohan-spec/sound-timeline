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

from clap_palette_score import PaletteScorer
from infer_reference_elements_timeline import load_audio


STEMS = ["vocals", "drums", "bass", "other"]


def segment_audio(path: Path, segment_seconds: float, hop_seconds: float) -> list[tuple[float, float, np.ndarray, int]]:
    audio, sr = load_audio(path)
    segment_len = max(1, int(segment_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    duration = len(audio) / sr
    starts = list(range(0, max(1, len(audio) - segment_len + 1), hop_len)) or [0]
    out = []
    for start in starts:
        end = min(len(audio), start + segment_len)
        clip = audio[start:end]
        if len(clip) < segment_len:
            pad = np.zeros((segment_len - len(clip), clip.shape[1]), dtype=np.float32)
            clip = np.vstack([clip, pad])
        out.append((round(start / sr, 4), round(min(duration, (start + segment_len) / sr), 4), clip, sr))
    return out


def stem_adjust(stem: str, item: dict) -> float:
    label = str(item["label"])
    family = str(item["family"])
    score = float(item["score"])
    if stem == "vocals":
        if family == "vocals":
            return score * 1.35
        if family == "drums":
            return score * 0.30
    if stem == "drums":
        if family == "drums":
            return score * 1.35
        if family in {"vocals", "guitar_strings_keys"}:
            return score * 0.45
    if stem == "bass":
        if family == "bass":
            return score * 1.45
        if family in {"vocals", "drums"}:
            return score * 0.45
    if stem == "other":
        if family in {"synth", "guitar_strings_keys", "sample_fx"}:
            return score * 1.15
    return score


def score_segment(scorer: PaletteScorer, stem: str, clip: np.ndarray, sr: int, clip_path: Path, top_k: int) -> list[dict]:
    sf.write(clip_path, clip, sr)
    report = scorer.score(clip_path)
    items = []
    for item in report["top_overall"]:
        adjusted = stem_adjust(stem, item)
        items.append({**item, "adjusted_score": adjusted})
    items.sort(key=lambda row: row["adjusted_score"], reverse=True)
    return items[:top_k]


def write_html(rows: list[dict], out_html: Path) -> None:
    counter = Counter()
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
        counter[row["primary_label"]] += 1

    count_body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
    sections = []
    for track, track_rows in sorted(by_track.items()):
        detail = []
        for row in track_rows:
            detail.append(
                "<tr>"
                f"<td>{html.escape(row['stem'])}</td>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(row['stem_path'])}'></audio></td>"
                f"<td>{html.escape(row['primary_label'])}</td>"
                f"<td>{row['primary_score']}</td>"
                f"<td>{html.escape(row['top_source_kinds'])}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(track)}</h2>
<table>
<tr><th>Stem</th><th>Time</th><th>Stem Audio</th><th>Primary CLAP Source Kind</th><th>Score</th><th>Top Source Kinds</th></tr>
{''.join(detail)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>CLAP Demucs Stem Source-Kind Analysis</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
</style>
<h1>CLAP Demucs Stem Source-Kind Analysis</h1>
<p>Open-source CLAP is used as a zero-shot text/audio panel over Demucs stems. Scores are prompt rankings, not probabilities. This is useful when local reference training samples are weak.</p>
<h2>Primary Label Counts</h2>
<table><tr><th>Label</th><th>Count</th></tr>{count_body}</table>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_test/htdemucs"))
    parser.add_argument("--palette", type=Path, default=Path("configs/source_kind_clap_prompts.json"))
    parser.add_argument("--model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_test/clap_stem_source_kind.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_test/clap_stem_source_kind.html"))
    args = parser.parse_args()

    scorer = PaletteScorer(args.palette, args.model_name)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        processed = 0
        for track_dir in sorted(path for path in args.stems_root.iterdir() if path.is_dir()):
            for stem in STEMS:
                stem_path = track_dir / f"{stem}.wav"
                if not stem_path.exists():
                    continue
                for idx, (start, end, clip, sr) in enumerate(segment_audio(stem_path, args.segment_seconds, args.hop_seconds)):
                    if args.max_segments and processed >= args.max_segments:
                        break
                    print(f"{track_dir.name} {stem} {start}-{end}s")
                    clip_path = tmp_dir / f"{track_dir.name}_{stem}_{idx:04d}.wav"
                    items = score_segment(scorer, stem, clip, sr, clip_path, args.top_k)
                    top = items[0] if items else {"label": "", "adjusted_score": 0.0}
                    processed += 1
                    rows.append(
                        {
                            "track": track_dir.name,
                            "stem": stem,
                            "start": start,
                            "end": end,
                            "stem_path": stem_path.as_posix(),
                            "primary_label": top["label"],
                            "primary_score": round(float(top["adjusted_score"]), 8),
                            "top_source_kinds": "; ".join(
                                f"{item['label']}:{float(item['adjusted_score']):.6f}" for item in items
                            ),
                        }
                    )
                if args.max_segments and processed >= args.max_segments:
                    break
            if args.max_segments and processed >= args.max_segments:
                break
    if not rows:
        raise SystemExit(f"No stem wav files found under {args.stems_root}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
