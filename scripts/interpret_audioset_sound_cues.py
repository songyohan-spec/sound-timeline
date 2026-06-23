from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


def parse_tags(value: str) -> list[tuple[str, float]]:
    tags = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, score = part.rsplit(":", 1)
        try:
            tags.append((label.strip(), float(score)))
        except ValueError:
            continue
    return tags


def has(tags: list[tuple[str, float]], words: list[str], min_score: float = 0.015) -> bool:
    for label, score in tags:
        lower = label.lower()
        if score >= min_score and any(word in lower for word in words):
            return True
    return False


def cue_score(tags: list[tuple[str, float]], words: list[str]) -> float:
    total = 0.0
    for label, score in tags:
        lower = label.lower()
        if any(word in lower for word in words):
            total += score
    return round(total, 4)


def interpret(tags: list[tuple[str, float]]) -> list[dict]:
    cues: list[dict] = []

    def add(label: str, family: str, reason_words: list[str], description: str) -> None:
        score = cue_score(tags, reason_words)
        if score > 0:
            cues.append(
                {
                    "family": family,
                    "cue": label,
                    "score": score,
                    "description": description,
                }
            )

    if has(tags, ["speech", "singing", "voice", "vocal", "rapping"]):
        add("vocal_presence", "vocals", ["speech", "singing", "voice", "vocal", "rapping"], "voice-like or vocal material is plausible")
    if has(tags, ["speech"]) and not has(tags, ["singing"], 0.02):
        add("spoken_or_processed_voice", "vocals", ["speech"], "voice-like material may be speech-like, chopped, or heavily processed")
    if has(tags, ["singing", "rapping"]):
        add("lead_or_hook_vocal", "vocals", ["singing", "rapping"], "sung or rap vocal presence is plausible")

    if has(tags, ["electronic music", "electronica", "techno", "dubstep", "sampler", "synth", "keyboard", "ringtone"]):
        add("electronic_synth_texture", "synth", ["electronic music", "electronica", "techno", "dubstep", "synth", "keyboard", "ringtone"], "electronic/synthetic production texture is plausible")
    if has(tags, ["sampler", "sample"]):
        add("sampled_or_resampled_loop", "sampled_loop", ["sampler", "sample"], "sampled or resampled loop character is plausible")
    if has(tags, ["ringtone", "ding", "jingle", "tinkle", "ping", "chink", "clink", "clang"]):
        add("bell_pluck_or_tiny_digital_hook", "synth", ["ringtone", "ding", "jingle", "tinkle", "ping", "chink", "clink", "clang"], "small bright bell/pluck/digital-hook character is plausible")
    if has(tags, ["dubstep", "drum and bass", "techno", "electronic dance"]):
        add("club_or_bass_music_influence", "bass", ["dubstep", "drum and bass", "techno", "electronic dance"], "club/bass-music influence is plausible, not necessarily an isolated bass stem")

    if has(tags, ["guitar", "strum", "plucked string"]):
        add("guitar_or_plucked_string", "guitar_strings", ["guitar", "strum", "plucked string"], "guitar or plucked-string-like source is plausible")
    if has(tags, ["rock", "metal", "grunge"]):
        add("rock_guitar_energy", "guitar_strings", ["rock", "metal", "grunge"], "rock/metal/grunge energy may indicate distorted guitar-like density")

    if has(tags, ["drum machine", "drum", "snare", "hi-hat", "percussion", "beat"]):
        add("drum_or_percussion_presence", "drums", ["drum machine", "drum", "snare", "hi-hat", "percussion", "beat"], "drum/percussion evidence is present")
    if has(tags, ["bang", "burst", "pop", "sound effect"]):
        add("hit_or_fx_transient", "noise_fx", ["bang", "burst", "pop", "sound effect"], "transient hit/fx evidence is present")
    if has(tags, ["static", "noise", "hum", "click", "whoosh"]):
        add("noise_bed_or_artifact", "noise_fx", ["static", "noise", "hum", "click", "whoosh"], "noise/artifact texture is plausible")

    cues.sort(key=lambda item: item["score"], reverse=True)
    return cues


def write_html(rows: list[dict], out_html: Path) -> None:
    family_counter: Counter[str] = Counter()
    cue_counter: Counter[str] = Counter()
    for row in rows:
        for cue in row["cue_items"]:
            family_counter[cue["family"]] += 1
            cue_counter[cue["cue"]] += 1

    def table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    detail = []
    for row in rows:
        cue_text = "<br>".join(
            f"<b>{html.escape(cue['cue'])}</b> ({cue['score']:.3f})<br><span class='muted'>{html.escape(cue['description'])}</span>"
            for cue in row["cue_items"][:6]
        ) or "-"
        detail.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload=\"metadata\" src=\"{html.escape(row.get('clip', ''))}\"></audio></td>"
            f"<td>{html.escape(row.get('teacher_confidence', ''))}</td>"
            f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
            f"<td>{cue_text}</td>"
            f"<td>{html.escape(row.get('audioset_top', '') or '-')}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AudioSet Sound Cue Interpreter</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 18px; }}
.muted {{ color: #555; font-size: 12px; }}
</style>
<h1>AudioSet Sound Cue Interpreter</h1>
<p>This translates public AudioSet/AST tags into sound-design-oriented cues. It is not source separation, but it is more informative than broad group names alone.</p>
<div class="grid">
{table("Cue Families", family_counter)}
{table("Cue Mentions", cue_counter)}
</div>
<h2>Segment Detail</h2>
<table>
<tr><th>Segment</th><th>Audio</th><th>Panel Decision</th><th>Project Labels</th><th>Interpreted Public Cues</th><th>Raw AudioSet Tags</th></tr>
{''.join(detail)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/review_queue/public_model_filtered.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/review_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/review_queue/audioset_sound_cues.html"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in rows:
        cues = interpret(parse_tags(row.get("audioset_top", "")))
        out = dict(row)
        out["public_sound_cues"] = "|".join(cue["cue"] for cue in cues[:6])
        out["public_sound_families"] = "|".join(dict.fromkeys(cue["family"] for cue in cues[:6]))
        out["public_sound_cue_scores"] = "; ".join(f"{cue['cue']}:{cue['score']:.3f}" for cue in cues[:6])
        out["cue_items"] = cues
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [key for key in out_rows[0].keys() if key != "cue_items"] if out_rows else []
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow({key: value for key, value in row.items() if key != "cue_items"})
    write_html(out_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
