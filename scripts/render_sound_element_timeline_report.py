from __future__ import annotations

import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


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


def parse_project_candidates(value: str, limit: int = 5) -> list[str]:
    out = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, rest = part.rsplit(":", 1)
        out.append(f"{label.strip()} ({rest.strip()})")
    return out[:limit]


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


def rank_cues(cues: list[tuple[str, float]]) -> list[tuple[str, float]]:
    def key(item: tuple[str, float]) -> tuple[float, float]:
        label, score = item
        penalty = -0.08 if label in GENERIC_CUES and len(cues) > 1 else 0.0
        return (score + SPECIFICITY_BONUS.get(label, 0.0) + penalty, score)

    return sorted(cues, key=key, reverse=True)


def primary_read(row: dict) -> tuple[str, str, float]:
    cues = rank_cues(parse_score_list(row.get("public_sound_cue_scores", "")))
    if cues:
        cue, score = cues[0]
        return cue, "public_ast", score
    labels = split_pipe(row.get("detected_labels", ""))
    if labels:
        return labels[0], "project_model", 0.0
    candidates = parse_project_candidates(row.get("top_labels", ""), 1)
    if candidates:
        return candidates[0], "project_possible", 0.0
    return "unclear", "none", 0.0


def confidence_class(row: dict, source: str, score: float) -> str:
    if row.get("teacher_confidence") == "public_supported":
        return "supported"
    if row.get("teacher_confidence") == "public_disagrees":
        return "disputed"
    if source == "public_ast" and score >= 0.08:
        return "public_hint"
    return "weak"


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    timeline_rows = []
    for row in rows:
        label, source, score = primary_read(row)
        cues = rank_cues(parse_score_list(row.get("public_sound_cue_scores", "")))
        secondary = [name for name, _ in cues[1:5]]
        project = parse_project_candidates(row.get("top_labels", ""), 5)
        cls = confidence_class(row, source, score)
        timeline_rows.append(
            {
                "file": row["file"],
                "start": row["start"],
                "end": row["end"],
                "clip": row.get("clip", ""),
                "primary_read": label,
                "primary_source": source,
                "primary_score": round(score, 4),
                "confidence": cls,
                "public_families": row.get("public_sound_families", ""),
                "supporting_public_cues": "|".join(secondary),
                "project_detected": row.get("detected_labels", ""),
                "project_candidates": "|".join(project),
                "kept_groups": row.get("teacher_filtered_groups", ""),
                "demoted_groups": row.get("teacher_suppressed_groups", ""),
                "raw_audioset": row.get("audioset_top", ""),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(timeline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(timeline_rows)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in timeline_rows:
        by_file[row["file"]].append(row)

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        trs = []
        for row in file_rows:
            supporting = "<br>".join(html.escape(x) for x in split_pipe(row["supporting_public_cues"])) or "-"
            project = "<br>".join(html.escape(x) for x in split_pipe(row["project_candidates"])) or "-"
            demoted = html.escape(row["demoted_groups"] or "-")
            kept = html.escape(row["kept_groups"] or "-")
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row['clip'])}\"></audio></td>"
                f"<td><span class=\"badge {html.escape(row['confidence'])}\">{html.escape(row['confidence'])}</span><br><b>{html.escape(row['primary_read'])}</b><br><span class=\"muted\">{html.escape(row['primary_source'])} {row['primary_score']}</span></td>"
                f"<td>{html.escape(row['public_families'] or '-')}</td>"
                f"<td>{supporting}</td>"
                f"<td>{html.escape(row['project_detected'] or '-')}</td>"
                f"<td>{project}</td>"
                f"<td><b>kept</b>: {kept}<br><b>demoted</b>: {demoted}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(file_name)}</h2>
<table>
<tr><th>Time</th><th>Audio</th><th>Primary Read</th><th>Public Families</th><th>Supporting Public Cues</th><th>Project Detected</th><th>Project Candidates</th><th>Cross-check</th></tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Sound Element Timeline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.badge {{ display:inline-block; padding:2px 6px; border-radius:4px; font-size:12px; font-weight:bold; margin-bottom:4px; }}
.supported {{ background:#dff3df; }}
.public_hint {{ background:#e4efff; }}
.disputed {{ background:#ffe1df; }}
.weak {{ background:#eee; }}
.muted {{ color:#555; font-size:12px; }}
</style>
<h1>Sound Element Timeline</h1>
<p>Primary Read favors public AST sound cues when available, then falls back to project-model candidates. Treat this as a ranked listening guide, not exact source separation.</p>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_element_timeline.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_element_timeline.html"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    write_outputs(rows, args.out_csv, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
