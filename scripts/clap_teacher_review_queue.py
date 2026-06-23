from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from clap_palette_score import PaletteScorer


LABEL_TO_FAMILY_HINTS = {
    "processed_lead_vocal": {"vocal_derived"},
    "hard_tuned_vocal": {"vocal_derived"},
    "pitched_vocal_chop": {"vocal_derived"},
    "breathy_vocal_pad": {"vocal_derived"},
    "stacked_harmony_vocal": {"vocal_derived"},
    "vocal_synth_hybrid": {"vocal_derived", "hybrid_sampled"},
    "lush_synth_pad": {"synth_derived"},
    "syrupy_video_game_synth_melody": {"synth_derived"},
    "bitcrushed_synth_lead": {"synth_derived"},
    "noisy_wavetable_texture": {"synth_derived", "fx_texture"},
    "glitch_percussion": {"fx_texture", "processing_space"},
    "trap_drum_pattern": {"drums"},
    "pulsing_sidechain_bass": {"processing_space", "bass"},
    "unknown_hybrid_loop": {"hybrid_sampled"},
    "filtered_sample_loop": {"hybrid_sampled"},
}


def parse_top_label(value: str) -> str:
    first = str(value or "").split(";", 1)[0].strip()
    if ":" not in first:
        return ""
    return first.rsplit(":", 1)[0].strip()


def top_clap_items(report: dict, top_k: int) -> list[dict]:
    return report.get("top_overall", [])[:top_k]


def clap_family_set(items: list[dict]) -> set[str]:
    return {str(item.get("family", "")) for item in items if item.get("family")}


def agreement(model_label: str, clap_items: list[dict]) -> str:
    if not model_label:
        return "no-model-label"
    hints = LABEL_TO_FAMILY_HINTS.get(model_label, set())
    if not hints:
        return "unknown-map"
    families = clap_family_set(clap_items[:5])
    if hints & families:
        return "support"
    return "disagree"


def write_html(rows: list[dict], path: Path) -> None:
    body = []
    for idx, row in enumerate(rows):
        body.append(
            "<tr>"
            f"<td>{idx + 1}</td>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row['clip'])}\"></audio></td>"
            f"<td>{html.escape(row['model_label'] or '-')}</td>"
            f"<td>{html.escape(row['clap_top'])}</td>"
            f"<td>{html.escape(row['teacher_agreement'])}</td>"
            f"<td>{html.escape(row.get('top_labels', ''))}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>CLAP Teacher Review Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 240px; }}
.note {{ color: #444; }}
</style>
<h1>CLAP Teacher Review Queue</h1>
<p class="note">CLAP is a semantic teacher panel, not ground truth. Use disagreement rows as review priorities.</p>
<table>
<tr><th>#</th><th>Segment</th><th>Audio</th><th>Our Top Label</th><th>CLAP Top</th><th>Agreement</th><th>Our Candidates</th></tr>
{''.join(body)}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("outputs/review_queue_detected/review_queue.csv"))
    parser.add_argument("--palette", type=Path, default=Path("configs/sound_palette_prompts.json"))
    parser.add_argument("--model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/review_queue_detected/clap_teacher.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/review_queue_detected/clap_teacher.html"))
    args = parser.parse_args()

    scorer = PaletteScorer(args.palette, args.model_name)
    with args.queue.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))[: args.limit]

    out_rows = []
    for idx, row in enumerate(rows, 1):
        clip_path = args.queue.parent / row["clip"]
        print(f"[{idx}/{len(rows)}] {clip_path}")
        report = scorer.score(clip_path)
        clap_items = top_clap_items(report, 5)
        model_label = row.get("detected_labels") or parse_top_label(row.get("top_labels", ""))
        out = dict(row)
        out["model_label"] = model_label
        out["clap_top"] = "; ".join(
            f"{item['label']}:{item['score']:.4f}" for item in clap_items
        )
        out["teacher_agreement"] = agreement(model_label, clap_items)
        out["clap_json"] = json.dumps(clap_items, ensure_ascii=False)
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
