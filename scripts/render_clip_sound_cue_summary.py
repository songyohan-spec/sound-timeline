from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


GENERIC_CUES = {"vocal_presence"}
SPECIFICITY_BONUS = {
    "lead_or_hook_vocal": 0.08,
    "spoken_or_processed_voice": 0.06,
    "electronic_synth_texture": 0.06,
    "bell_pluck_or_tiny_digital_hook": 0.06,
    "guitar_or_plucked_string": 0.06,
    "club_or_bass_music_influence": 0.05,
    "hit_or_fx_transient": 0.04,
    "noise_bed_or_artifact": 0.04,
    "drum_or_percussion_presence": 0.04,
    "sampled_or_resampled_loop": 0.04,
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


def rank_cues(cues: list[tuple[str, float]]) -> list[tuple[str, float]]:
    def key(item: tuple[str, float]) -> tuple[float, float]:
        label, score = item
        penalty = -0.08 if label in GENERIC_CUES and len(cues) > 1 else 0.0
        return (score + SPECIFICITY_BONUS.get(label, 0.0) + penalty, score)

    return sorted(cues, key=key, reverse=True)


def top_items(counter: Counter[str], limit: int = 8) -> str:
    if not counter:
        return "-"
    return "<br>".join(f"{html.escape(label)} <span class='count'>{count}</span>" for label, count in counter.most_common(limit))


def write_html(rows: list[dict], out_html: Path) -> None:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["file"]].append(row)

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        family_counter: Counter[str] = Counter()
        cue_counter: Counter[str] = Counter()
        project_counter: Counter[str] = Counter()
        kept_counter: Counter[str] = Counter()
        demoted_counter: Counter[str] = Counter()
        detail = []
        for row in file_rows:
            family_counter.update(split_pipe(row.get("public_sound_families", "")))
            cue_counter.update([label for label, _ in rank_cues(parse_score_list(row.get("public_sound_cue_scores", "")))[:4]])
            project_counter.update(split_pipe(row.get("detected_labels", "")))
            kept_counter.update(split_pipe(row.get("teacher_filtered_groups", "")))
            demoted_counter.update(split_pipe(row.get("teacher_suppressed_groups", "")))
            detail.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row.get('clip', ''))}\"></audio></td>"
                f"<td>{html.escape(row.get('public_sound_cue_scores', '') or '-')}</td>"
                f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
                f"<td>{html.escape(row.get('teacher_filtered_groups', '') or '-')}</td>"
                f"<td>{html.escape(row.get('teacher_suppressed_groups', '') or '-')}</td>"
                "</tr>"
            )

        sections.append(
            f"""<section class="clip">
<h2>{html.escape(file_name)}</h2>
<div class="grid">
<div><h3>Public Cue Families</h3>{top_items(family_counter)}</div>
<div><h3>Public Sound Cues</h3>{top_items(cue_counter)}</div>
<div><h3>Project Labels</h3>{top_items(project_counter)}</div>
<div><h3>Kept / Demoted</h3><b>Kept</b><br>{top_items(kept_counter, 5)}<br><br><b>Demoted</b><br>{top_items(demoted_counter, 5)}</div>
</div>
<table>
<tr><th>Time</th><th>Audio</th><th>Public Cue Scores</th><th>Project Labels</th><th>Kept Groups</th><th>Demoted Groups</th></tr>
{''.join(detail)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Clip Sound Cue Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
.clip {{ border-top: 2px solid #111; padding-top: 14px; margin-top: 28px; }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 18px; }}
table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.count {{ color: #666; font-size: 12px; }}
h3 {{ margin-bottom: 6px; }}
</style>
<h1>Clip Sound Cue Summary</h1>
<p>Per-clip summary from the full 2-second segment pass. Public cues come from AudioSet/AST tag interpretation; project labels are synthetic-reference model candidates.</p>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/clip_sound_cue_summary.html"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
