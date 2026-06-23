from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


VOCAL_LABELS = [
    "lead_or_hook_vocal",
    "spoken_processed_vocal",
    "hard_tuned_vocal",
    "pitched_vocal_chop",
    "breathy_vocal_pad",
    "stacked_harmony",
    "vocal_synth_hybrid",
    "vocoder_or_synthetic_vocal",
]

SYNTH_LABELS = [
    "synth_pad_or_wash",
    "digital_pluck_or_bell",
    "bitcrushed_synth_lead",
    "noisy_wavetable_texture",
    "game_like_synth_melody",
    "filtered_sample_or_synth_loop",
    "ambient_electronic_texture",
    "bass_synth_pulse",
]


def parse_score_list(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, score = part.rsplit(":", 1)
        try:
            out[label.strip()] = float(score)
        except ValueError:
            continue
    return out


def parse_project_top(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, rest = part.rsplit(":", 1)
        score = rest.split("/", 1)[0].strip()
        try:
            out[label.strip()] = float(score)
        except ValueError:
            continue
    return out


def audioset_score(tags: dict[str, float], words: list[str]) -> float:
    total = 0.0
    for label, score in tags.items():
        lower = label.lower()
        if any(word in lower for word in words):
            total += score
    return total


def dsp(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def detail_scores(row: dict) -> tuple[dict[str, float], dict[str, float]]:
    project = parse_project_top(row.get("top_labels", ""))
    public_cues = parse_score_list(row.get("public_sound_cue_scores", ""))
    audio_tags = parse_score_list(row.get("audioset_top", ""))
    centroid = dsp(row, "brightness_centroid")
    flatness = dsp(row, "flatness_noise")
    width = dsp(row, "width")

    vocals = {label: 0.0 for label in VOCAL_LABELS}
    synths = {label: 0.0 for label in SYNTH_LABELS}

    vocals["lead_or_hook_vocal"] = max(
        public_cues.get("lead_or_hook_vocal", 0.0),
        audioset_score(audio_tags, ["singing", "vocal music", "female singing", "male singing", "rapping"]),
        project.get("processed_lead_vocal", 0.0) * 0.45,
    )
    vocals["spoken_processed_vocal"] = max(
        public_cues.get("spoken_or_processed_voice", 0.0),
        audioset_score(audio_tags, ["speech", "voice"]) * 0.9,
    )
    vocals["hard_tuned_vocal"] = max(
        project.get("hard_tuned_vocal", 0.0) * 0.65,
        project.get("processed_lead_vocal", 0.0) * 0.28 if centroid > 1200 else 0.0,
    )
    vocals["pitched_vocal_chop"] = max(
        project.get("pitched_vocal_chop", 0.0) * 0.65,
        project.get("vocal_synth_hybrid", 0.0) * 0.25,
    )
    vocals["breathy_vocal_pad"] = max(
        project.get("breathy_vocal_pad", 0.0) * 0.70,
        project.get("stacked_harmony_vocal", 0.0) * 0.25 if width > 0.7 else 0.0,
    )
    vocals["stacked_harmony"] = max(
        project.get("stacked_harmony_vocal", 0.0) * 0.70,
        audioset_score(audio_tags, ["choir", "chorus"]) * 0.8,
    )
    vocals["vocal_synth_hybrid"] = max(
        project.get("vocal_synth_hybrid", 0.0) * 0.70,
        min(public_cues.get("lead_or_hook_vocal", 0.0), public_cues.get("electronic_synth_texture", 0.0)) * 1.2,
    )
    vocals["vocoder_or_synthetic_vocal"] = max(
        project.get("vocoder_vocal_texture", 0.0) * 0.75,
        audioset_score(audio_tags, ["synthetic singing"]) * 1.1,
    )

    synths["synth_pad_or_wash"] = max(
        project.get("lush_synth_pad", 0.0) * 0.70,
        public_cues.get("electronic_synth_texture", 0.0) * 0.55 if width > 0.55 else 0.0,
    )
    synths["digital_pluck_or_bell"] = max(
        public_cues.get("bell_pluck_or_tiny_digital_hook", 0.0),
        audioset_score(audio_tags, ["bell", "ringtone", "ding", "jingle", "tinkle", "ping"]),
        project.get("glitching_bell_texture", 0.0) * 0.70,
    )
    synths["bitcrushed_synth_lead"] = max(
        project.get("bitcrushed_synth_lead", 0.0) * 0.72,
        project.get("fuzzy_diy_synth_texture", 0.0) * 0.50,
    )
    synths["noisy_wavetable_texture"] = max(
        project.get("noisy_wavetable_texture", 0.0) * 0.72,
        public_cues.get("electronic_synth_texture", 0.0) * 0.35 if flatness > 0.16 else 0.0,
    )
    synths["game_like_synth_melody"] = max(
        project.get("syrupy_video_game_synth_melody", 0.0) * 0.72,
        public_cues.get("bell_pluck_or_tiny_digital_hook", 0.0) * 0.45,
    )
    synths["filtered_sample_or_synth_loop"] = max(
        project.get("filtered_sample_loop", 0.0) * 0.65,
        project.get("unknown_hybrid_loop", 0.0) * 0.40,
    )
    synths["ambient_electronic_texture"] = max(
        public_cues.get("electronic_synth_texture", 0.0),
        audioset_score(audio_tags, ["electronic music", "electronica", "ambient", "background music"]) * 0.8,
    )
    synths["bass_synth_pulse"] = max(
        project.get("pulsing_sidechain_bass", 0.0) * 0.55,
        audioset_score(audio_tags, ["dubstep", "drum and bass", "bass"]) * 0.45,
    )

    return ({key: round(value, 5) for key, value in vocals.items()}, {key: round(value, 5) for key, value in synths.items()})


def active(scores: dict[str, float], threshold: float = 0.10, limit: int = 3) -> list[str]:
    return [
        label
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= threshold
    ][:limit]


def bar(score: float) -> str:
    width = min(100, int(score * 300))
    if score >= 0.18:
        cls = "strong"
    elif score >= 0.09:
        cls = "medium"
    elif score >= 0.10:
        cls = "weak"
    else:
        cls = "none"
    return f"<div class='bar {cls}'><span style='width:{width}%'></span></div><small>{score:.3f}</small>"


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    out_rows = []
    vocal_counter: Counter[str] = Counter()
    synth_counter: Counter[str] = Counter()
    for row in rows:
        vocals, synths = detail_scores(row)
        vocal_active = active(vocals)
        synth_active = active(synths)
        vocal_counter.update(vocal_active)
        synth_counter.update(synth_active)
        out = {
            "file": row["file"],
            "start": row["start"],
            "end": row["end"],
            "clip": row.get("clip", ""),
            "vocal_candidates": "|".join(vocal_active),
            "synth_candidates": "|".join(synth_active),
        }
        out.update({f"vocal_{key}": value for key, value in vocals.items()})
        out.update({f"synth_{key}": value for key, value in synths.items()})
        out_rows.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        by_file[row["file"]].append(row)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Candidate</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        trs = []
        for row in file_rows:
            vocal_cells = "<br>".join(f"{label}: {bar(float(row[f'vocal_{label}']))}" for label in VOCAL_LABELS)
            synth_cells = "<br>".join(f"{label}: {bar(float(row[f'synth_{label}']))}" for label in SYNTH_LABELS)
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
                f"<td>{html.escape(row['vocal_candidates'] or '-')}</td>"
                f"<td>{html.escape(row['synth_candidates'] or '-')}</td>"
                f"<td>{vocal_cells}</td>"
                f"<td>{synth_cells}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(file_name)}</h2>
<table>
<tr><th>Time</th><th>Audio</th><th>Vocal Candidates</th><th>Synth Candidates</th><th>Vocal Scores</th><th>Synth Scores</th></tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Vocal / Synth Detail</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
.bar {{ display:inline-block; width:70px; height:8px; background:#eee; margin:0 4px; }}
.bar span {{ display:block; height:8px; background:#999; }}
.bar.strong span {{ background:#255c9b; }}
.bar.medium span {{ background:#6298cd; }}
.bar.weak span {{ background:#a8c3dc; }}
small {{ color:#555; }}
</style>
<h1>Vocal / Synth Detail</h1>
<p>Heuristic detail layer for the current reference set. It combines public AST cues, project candidates, and simple DSP hints. Treat as ranked hypotheses.</p>
<div>
{count_table("Vocal Candidate Mentions", vocal_counter)}
{count_table("Synth Candidate Mentions", synth_counter)}
</div>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/vocal_synth_detail.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/vocal_synth_detail.html"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    write_outputs(rows, args.out_csv, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
