from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


FAMILIES = ["vocals", "synth", "guitar_strings", "bass", "drums", "noise_fx", "sampled_loop"]

CUE_TO_FAMILY = {
    "lead_or_hook_vocal": "vocals",
    "spoken_or_processed_voice": "vocals",
    "vocal_presence": "vocals",
    "electronic_synth_texture": "synth",
    "bell_pluck_or_tiny_digital_hook": "synth",
    "guitar_or_plucked_string": "guitar_strings",
    "rock_guitar_energy": "guitar_strings",
    "club_or_bass_music_influence": "bass",
    "drum_or_percussion_presence": "drums",
    "hit_or_fx_transient": "noise_fx",
    "noise_bed_or_artifact": "noise_fx",
    "sampled_or_resampled_loop": "sampled_loop",
}

PROJECT_LABEL_TO_FAMILY = {
    "processed_lead_vocal": "vocals",
    "hard_tuned_vocal": "vocals",
    "pitched_vocal_chop": "vocals",
    "breathy_vocal_pad": "vocals",
    "stacked_harmony_vocal": "vocals",
    "vocal_synth_hybrid": "vocals",
    "vocoder_vocal_texture": "vocals",
    "lush_synth_pad": "synth",
    "syrupy_video_game_synth_melody": "synth",
    "bitcrushed_synth_lead": "synth",
    "noisy_wavetable_texture": "synth",
    "glitching_bell_texture": "synth",
    "pulsing_sidechain_bass": "bass",
    "sub_bass": "bass",
    "distorted_808_bass": "bass",
    "trap_drum_pattern": "drums",
    "glitch_percussion": "drums",
    "trap_hi_hat_rolls": "drums",
    "electronic_clap_snare": "drums",
    "filtered_guitar_loop": "guitar_strings",
    "washed_chorus_guitar": "guitar_strings",
    "distorted_guitar_texture": "guitar_strings",
    "unknown_hybrid_loop": "sampled_loop",
    "filtered_sample_loop": "sampled_loop",
}


def parse_score_list(value: str) -> list[tuple[str, float]]:
    items = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, score = part.rsplit(":", 1)
        try:
            items.append((label.strip(), float(score)))
        except ValueError:
            continue
    return items


def parse_project_top(value: str) -> list[tuple[str, float]]:
    items = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, rest = part.rsplit(":", 1)
        score_text = rest.split("/", 1)[0].strip()
        try:
            items.append((label.strip(), float(score_text)))
        except ValueError:
            continue
    return items


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def family_scores(row: dict) -> dict[str, float]:
    scores = {family: 0.0 for family in FAMILIES}
    for cue, score in parse_score_list(row.get("public_sound_cue_scores", "")):
        family = CUE_TO_FAMILY.get(cue)
        if family:
            scores[family] = max(scores[family], score)

    # Project model contributes much weaker evidence because it is synthetic-reference trained.
    for label, score in parse_project_top(row.get("top_labels", ""))[:6]:
        family = PROJECT_LABEL_TO_FAMILY.get(label)
        if family and score >= 0.30:
            scores[family] = max(scores[family], score * 0.18)

    for family in split_pipe(row.get("teacher_filtered_groups", "")):
        if family in scores:
            scores[family] = max(scores[family], 0.18)
    for family in split_pipe(row.get("teacher_suppressed_groups", "")):
        if family in scores:
            scores[family] *= 0.20
    return scores


def strength(score: float) -> str:
    if score >= 0.18:
        return "strong"
    if score >= 0.08:
        return "medium"
    if score >= 0.055:
        return "weak"
    return ""


def bar(score: float) -> str:
    width = min(100, int(score * 360))
    cls = strength(score) or "none"
    return f"<div class='bar {cls}'><span style='width:{width}%'></span></div><small>{score:.3f}</small>"


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    matrix_rows = []
    for row in rows:
        scores = family_scores(row)
        active = [family for family, value in sorted(scores.items(), key=lambda x: x[1], reverse=True) if value >= 0.055]
        matrix_row = {
            "file": row["file"],
            "start": row["start"],
            "end": row["end"],
            "clip": row.get("clip", ""),
            "active_layers": "|".join(active),
            "primary_read": row.get("primary_read", ""),
            "confidence": row.get("confidence", ""),
        }
        for family in FAMILIES:
            matrix_row[family] = round(scores[family], 5)
            matrix_row[f"{family}_strength"] = strength(scores[family])
        matrix_rows.append(matrix_row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(matrix_rows[0].keys()))
        writer.writeheader()
        writer.writerows(matrix_rows)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in matrix_rows:
        by_file[row["file"]].append(row)

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        trs = []
        for row in file_rows:
            cells = "".join(f"<td>{bar(float(row[family]))}</td>" for family in FAMILIES)
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
                f"<td>{html.escape(row['active_layers'] or '-')}</td>"
                f"{cells}"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(file_name)}</h2>
<table>
<tr><th>Time</th><th>Audio</th><th>Active Layers</th>{''.join(f'<th>{family}</th>' for family in FAMILIES)}</tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Sound Layer Matrix</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 190px; }}
.bar {{ width: 86px; height: 9px; background: #eee; margin-bottom: 2px; }}
.bar span {{ display: block; height: 9px; background: #999; }}
.bar.strong span {{ background: #2764c5; }}
.bar.medium span {{ background: #5a8fd8; }}
.bar.weak span {{ background: #a8bfdc; }}
small {{ color: #555; }}
</style>
<h1>Sound Layer Matrix</h1>
<p>Layer strength combines interpreted public AudioSet cues with weaker project-model evidence. This is a listening guide for co-occurring layers, not isolated stem recovery.</p>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_element_timeline.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_layer_matrix.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_layer_matrix.html"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        cue_rows = list(csv.DictReader(f))
    timeline_by_key = {}
    if args.timeline.exists():
        with args.timeline.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                timeline_by_key[(row["file"], row["start"], row["end"])] = row
    rows = []
    for row in cue_rows:
        merged = dict(row)
        merged.update(timeline_by_key.get((row["file"], row["start"], row["end"]), {}))
        rows.append(merged)
    write_outputs(rows, args.out_csv, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
