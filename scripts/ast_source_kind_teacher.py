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


def add(scores: dict[str, float], label: str, value: float) -> None:
    scores[label] = max(scores.get(label, 0.0), value)


def ast_source_kind_scores(tags: list[tuple[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for tag, score in tags:
        lower = tag.lower()
        if any(word in lower for word in ["singing", "vocal music", "female singing", "male singing"]):
            add(scores, "clean_or_lead_vocal", score)
            add(scores, "lead_or_hook_vocal", score * 0.9)
        if any(word in lower for word in ["speech", "rapping", "narration"]):
            add(scores, "rap_or_spoken_vocal", score)
        if any(word in lower for word in ["choir", "chorus"]):
            add(scores, "vocal_pad_or_harmony", score)
        if any(word in lower for word in ["synthetic singing"]):
            add(scores, "formant_or_vocoder_vocal", score)
        if any(word in lower for word in ["synthesizer", "electronic music", "electronica", "keyboard"]):
            add(scores, "synth_pad_or_wash", score * 0.62)
            add(scores, "digital_synth_lead", score * 0.42)
        if any(word in lower for word in ["ringtone", "bell", "ding", "jingle", "tinkle", "ping", "chink", "clink"]):
            add(scores, "synth_pluck_or_bell", score)
        if any(word in lower for word in ["sampler", "sample"]):
            add(scores, "sampled_loop_texture", score)
        if any(word in lower for word in ["bass", "sub-bass", "dubstep", "drum and bass"]):
            add(scores, "sub_or_808_bass", score * 0.7)
            add(scores, "synth_bass", score * 0.58)
        if any(word in lower for word in ["drum machine", "beat"]):
            add(scores, "electronic_drum_machine", score)
        if any(word in lower for word in ["drum kit", "breakbeat", "drum roll", "percussion"]):
            add(scores, "breakbeat_or_live_drums", score)
        if any(word in lower for word in ["bass drum", "kick drum", "thump", "bang"]):
            add(scores, "kick_or_low_hit", score)
        if any(word in lower for word in ["snare", "clap", "snap"]):
            add(scores, "snare_clap_or_snap", score)
        if any(word in lower for word in ["hi-hat", "cymbal", "tick", "click"]):
            add(scores, "hat_tick_or_click", score)
        if any(word in lower for word in ["guitar", "strum", "plucked string"]):
            add(scores, "guitar_or_plucked_loop", score)
        if any(word in lower for word in ["rock", "grunge", "metal", "electric guitar", "distortion"]):
            add(scores, "distorted_guitar_or_rock_texture", score * 0.65)
        if any(word in lower for word in ["violin", "string", "cello"]):
            add(scores, "string_or_violin_like", score)
        if any(word in lower for word in ["piano", "electric piano", "organ"]):
            add(scores, "piano_or_keyboard_loop", score)
            add(scores, "warm_keys_or_organ", score * 0.85)
        if any(word in lower for word in ["noise", "static", "sound effect", "whoosh", "burst", "explosion"]):
            add(scores, "noise_or_fx_transition", score)
        if any(word in lower for word in ["whoosh", "wind"]):
            add(scores, "riser_or_swell", score)
        if any(word in lower for word in ["bang", "burst", "explosion", "thump"]):
            add(scores, "impact_or_tail", score)
    return scores


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def load_heuristic(path: Path) -> dict[tuple[str, str, str], dict]:
    out = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            out[(row["file"], str(row["start"]), str(row["end"]))] = row
    return out


def format_scores(scores: dict[str, float], threshold: float, limit: int) -> str:
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return "; ".join(f"{label}:{score:.3f}{'/detected' if score >= threshold else '/possible'}" for label, score in items[:limit])


def write_html(rows: list[dict], out_html: Path) -> None:
    agreement_counter = Counter(row["agreement"] for row in rows)
    detected_counter = Counter()
    for row in rows:
        detected_counter.update(split_pipe(row["ast_detected"]))

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    detail = []
    for row in rows:
        detail.append(
            "<tr>"
            f"<td>{html.escape(row['file'])}<br>{row['start']}-{row['end']}s</td>"
            f"<td><audio controls preload='metadata' src='{html.escape(row.get('clip', ''))}'></audio></td>"
            f"<td>{html.escape(row['heuristic_active'] or '-')}</td>"
            f"<td>{html.escape(row['ast_detected'] or '-')}</td>"
            f"<td>{html.escape(row['overlap'] or '-')}</td>"
            f"<td>{html.escape(row['agreement'])}</td>"
            f"<td>{html.escape(row['ast_source_kind_top'])}</td>"
            f"<td>{html.escape(row.get('audioset_top', '') or '-')}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>AST Source Kind Teacher</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
</style>
<h1>AST Source Kind Teacher</h1>
<p>This is the open-source AST/AudioSet panel only, mapped into the source-kind vocabulary. It does not use the project-trained source-kind model.</p>
{count_table("Agreement With Current Source Kind Report", agreement_counter)}
{count_table("AST Detected Source Kinds", detected_counter)}
<h2>Segment Detail</h2>
<table>
<tr><th>Segment</th><th>Audio</th><th>Current Source Kinds</th><th>AST Source Kinds</th><th>Overlap</th><th>Agreement</th><th>AST Mapped Scores</th><th>Raw AST Tags</th></tr>
{''.join(detail)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--heuristic", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.csv"))
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/ast_source_kind_teacher.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/ast_source_kind_teacher.html"))
    args = parser.parse_args()

    heuristic = load_heuristic(args.heuristic)
    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    out_rows = []
    for row in rows:
        scores = ast_source_kind_scores(parse_tags(row.get("audioset_top", "")))
        detected = [label for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score >= args.threshold]
        key = (row["file"], str(row["start"]), str(row["end"]))
        heuristic_active = split_pipe(heuristic.get(key, {}).get("active_source_kinds", ""))
        overlap = sorted(set(detected) & set(heuristic_active))
        if overlap:
            agreement = "overlap"
        elif detected and heuristic_active:
            agreement = "disagree"
        elif detected:
            agreement = "ast_only"
        elif heuristic_active:
            agreement = "heuristic_only"
        else:
            agreement = "empty"
        out = dict(row)
        out["heuristic_active"] = "|".join(heuristic_active)
        out["ast_detected"] = "|".join(detected)
        out["overlap"] = "|".join(overlap)
        out["agreement"] = agreement
        out["ast_source_kind_top"] = format_scores(scores, args.threshold, args.top_k)
        out_rows.append(out)

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
