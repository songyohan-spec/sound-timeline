from __future__ import annotations

import argparse
import csv
import html
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clap_palette_score import PaletteScorer
from infer_reference_elements_timeline import load_audio


ACTIVE_STRENGTHS = {"medium", "strong"}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_segment(stem_path: Path, start: float, end: float, out_path: Path) -> None:
    audio, sr = load_audio(stem_path, sample_rate=48_000)
    start_i = max(0, int(start * sr))
    end_i = min(audio.shape[0], int(end * sr))
    clip = audio[start_i:end_i]
    if clip.shape[0] < int(0.25 * sr):
        raise ValueError("segment too short")
    sf.write(out_path, clip, sr)


def top_items(report: dict, limit: int) -> list[dict]:
    return list(report.get("top_overall", []))[:limit]


def top_non_negative(items: list[dict]) -> dict | None:
    for item in items:
        if item.get("family") != "not_synth" and not str(item.get("label", "")).startswith("not_synth"):
            return item
    return items[0] if items else None


def agreement(specialist_label: str, specialist_family: str, clap_item: dict | None, clap_items: list[dict]) -> tuple[str, str]:
    if not clap_item:
        return "no_clap", "CLAP returned no item"
    clap_label = str(clap_item["label"])
    clap_family = str(clap_item["family"])
    if clap_label == specialist_label:
        return "agree", "same label"
    if specialist_family == clap_family:
        return "family_agree", "same broad family"
    if clap_family == "not_synth":
        return "disagree", "CLAP top says not_synth"
    top_labels = {str(item.get("label", "")) for item in clap_items[:5]}
    if specialist_label in top_labels:
        return "soft_agree", "specialist label appears in CLAP top-5"
    return "disagree", "different synth interpretation"


def write_html(rows: list[dict], out_html: Path) -> None:
    agreement_counts = Counter(row["agreement"] for row in rows)
    label_counts = Counter(row["final_label"] for row in rows)
    count_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in agreement_counts.most_common()
    )
    label_rows = "".join(
        f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>"
        for label, count in label_counts.most_common(16)
    )
    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td>{html.escape(row['specialist_label'])} ({row['specialist_conf']})</td>"
            f"<td>{html.escape(row['clap_label'])} ({row['clap_score']})</td>"
            f"<td class='{html.escape(row['agreement'])}'>{html.escape(row['agreement'])}</td>"
            f"<td>{html.escape(row['final_label'])}</td>"
            f"<td>{html.escape(row['clap_top'])}</td>"
            f"<td><audio controls preload='none' src='{html.escape(row['stem_path'])}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>CLAP Synth Teacher Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.agree, .soft_agree, .family_agree {{ background: #e8f7ed; }}
.disagree {{ background: #ffe9dc; }}
.ambiguous, .no_clap {{ background: #fff7df; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>CLAP Synth Teacher Ensemble</h1>
<p class="note">Cross-checks the local synth specialist with an open-vocabulary CLAP prompt panel. Use agree/family_agree rows as stronger pseudo-labels; use disagreement rows as training-gap clues.</p>
<h2>Agreement</h2>
<table><tr><th>Agreement</th><th>Count</th></tr>{count_rows}</table>
<h2>Final Label Counts</h2>
<table><tr><th>Final Label</th><th>Count</th></tr>{label_rows}</table>
<h2>Details</h2>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Specialist</th><th>CLAP</th><th>Agreement</th><th>Final Label</th><th>CLAP Top</th><th>Audio</th></tr>
{''.join(detail_rows)}
</table>
<p class="note">Caution: CLAP scores are prompt rankings, not calibrated probabilities. This panel is for agreement filtering, not final truth.</p>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist.csv"))
    parser.add_argument("--palette", type=Path, default=Path("configs/synth_specialist_clap_prompts.json"))
    parser.add_argument("--model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--include-weak", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_ensemble.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_ensemble.html"))
    args = parser.parse_args()

    rows = []
    for row in read_rows(args.input):
        if row.get("strength") in ACTIVE_STRENGTHS or (args.include_weak and row.get("strength") == "weak"):
            rows.append(row)
    if args.max_rows:
        rows = rows[: args.max_rows]
    if not rows:
        raise SystemExit("No rows selected for CLAP teacher ensemble.")

    scorer = PaletteScorer(args.palette, args.model_name)
    out_rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for idx, row in enumerate(rows, 1):
            print(f"{idx}/{len(rows)} {row['track']} {row['stem']} {row['start']}-{row['end']}")
            clip_path = tmp_dir / f"clip_{idx:05d}.wav"
            try:
                write_segment(Path(row["stem_path"]), float(row["start"]), float(row["end"]), clip_path)
                report = scorer.score(clip_path)
                items = top_items(report, 8)
                clap_item = top_non_negative(items)
                status, reason = agreement(row["synth_label_top"], row["synth_family_top"], clap_item, items)
                clap_label = str(clap_item["label"]) if clap_item else ""
                clap_family = str(clap_item["family"]) if clap_item else ""
                clap_score = float(clap_item["score"]) if clap_item else 0.0
            except Exception as exc:
                items = []
                status, reason = "no_clap", type(exc).__name__
                clap_label, clap_family, clap_score = "", "", 0.0

            final_label = row["synth_label_top"] if status in {"agree", "soft_agree", "family_agree"} else "ambiguous"
            out_rows.append(
                {
                    "track": row["track"],
                    "stem": row["stem"],
                    "start": row["start"],
                    "end": row["end"],
                    "stem_path": row["stem_path"],
                    "specialist_family": row["synth_family_top"],
                    "specialist_label": row["synth_label_top"],
                    "specialist_conf": row["synth_label_conf"],
                    "specialist_strength": row["strength"],
                    "clap_family": clap_family,
                    "clap_label": clap_label,
                    "clap_score": round(clap_score, 8),
                    "agreement": status,
                    "agreement_reason": reason,
                    "final_label": final_label,
                    "clap_top": "; ".join(f"{item['family']}/{item['label']}:{float(item['score']):.6f}" for item in items[:5]),
                }
            )

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
