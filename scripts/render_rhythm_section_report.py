from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


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


def parse_project_top(value: str) -> dict[str, float]:
    out = {}
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


def audioset_score(tags: list[tuple[str, float]], words: list[str]) -> float:
    score = 0.0
    for label, value in tags:
        lower = label.lower()
        if any(word in lower for word in words):
            score += value
    return score


def f(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def rhythm_scores(row: dict) -> dict[str, float]:
    tags = parse_score_list(row.get("audioset_top", ""))
    project = parse_project_top(row.get("top_labels", ""))
    centroid = f(row, "brightness_centroid")
    flatness = f(row, "flatness_noise")
    motion = f(row, "motion_strength")

    scores = {
        "kick_or_low_hit": 0.0,
        "808_or_sub_bass": 0.0,
        "snare_or_clap": 0.0,
        "hi_hat_or_tick": 0.0,
        "glitch_percussion": 0.0,
        "breakbeat_or_drum_loop": 0.0,
        "sidechain_or_pumping": 0.0,
    }

    scores["kick_or_low_hit"] += audioset_score(tags, ["bang", "thump", "drum machine"]) * 0.65
    scores["808_or_sub_bass"] += audioset_score(tags, ["bass"]) * 0.65
    scores["snare_or_clap"] += audioset_score(tags, ["snare", "clap", "rimshot", "pop", "burst"]) * 0.9
    scores["hi_hat_or_tick"] += audioset_score(tags, ["hi-hat", "cymbal", "tick", "click"]) * 0.9
    scores["glitch_percussion"] += audioset_score(tags, ["static", "click", "sound effect", "glitch", "scratch"]) * 0.8
    scores["breakbeat_or_drum_loop"] += audioset_score(tags, ["beat", "breakbeat", "drum machine"]) * 0.55

    if "trap_drum_pattern" in project:
        if project["trap_drum_pattern"] >= 0.45:
            scores["breakbeat_or_drum_loop"] = max(scores["breakbeat_or_drum_loop"], project["trap_drum_pattern"] * 0.36)
            scores["kick_or_low_hit"] = max(scores["kick_or_low_hit"], project["trap_drum_pattern"] * 0.18)
    if "glitch_percussion" in project:
        scores["glitch_percussion"] = max(scores["glitch_percussion"], project["glitch_percussion"] * 0.55)
    if "trap_hi_hat_rolls" in project:
        scores["hi_hat_or_tick"] = max(scores["hi_hat_or_tick"], project["trap_hi_hat_rolls"] * 0.55)
    if "electronic_clap_snare" in project:
        scores["snare_or_clap"] = max(scores["snare_or_clap"], project["electronic_clap_snare"] * 0.55)
    if "pulsing_sidechain_bass" in project:
        if project["pulsing_sidechain_bass"] >= 0.45:
            scores["808_or_sub_bass"] = max(scores["808_or_sub_bass"], project["pulsing_sidechain_bass"] * 0.32)
            scores["sidechain_or_pumping"] = max(scores["sidechain_or_pumping"], project["pulsing_sidechain_bass"] * 0.38)
    if "distorted_808_bass" in project:
        scores["808_or_sub_bass"] = max(scores["808_or_sub_bass"], project["distorted_808_bass"] * 0.55)
    if "sub_bass" in project:
        scores["808_or_sub_bass"] = max(scores["808_or_sub_bass"], project["sub_bass"] * 0.50)

    # DSP hints: these are weak, but useful when public labels are generic.
    if centroid > 3500 and flatness > 0.18:
        scores["hi_hat_or_tick"] = max(scores["hi_hat_or_tick"], min(0.18, flatness * 0.35))
    if centroid > 1800 and flatness > 0.25:
        scores["glitch_percussion"] = max(scores["glitch_percussion"], min(0.20, flatness * 0.45))
    if motion > 0.22:
        scores["sidechain_or_pumping"] = max(scores["sidechain_or_pumping"], min(0.12, motion * 0.35))
    if centroid < 800 and motion > 0.08:
        scores["kick_or_low_hit"] = max(scores["kick_or_low_hit"], min(0.10, motion * 0.26))

    return {key: round(value, 5) for key, value in scores.items()}


def strength(score: float) -> str:
    if score >= 0.18:
        return "strong"
    if score >= 0.09:
        return "medium"
    if score >= 0.045:
        return "weak"
    return ""


def active(scores: dict[str, float]) -> list[str]:
    return [label for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True) if score >= 0.045]


def bar(score: float) -> str:
    width = min(100, int(score * 360))
    cls = strength(score) or "none"
    return f"<div class='bar {cls}'><span style='width:{width}%'></span></div><small>{score:.3f}</small>"


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    out_rows = []
    labels = [
        "kick_or_low_hit",
        "808_or_sub_bass",
        "snare_or_clap",
        "hi_hat_or_tick",
        "glitch_percussion",
        "breakbeat_or_drum_loop",
        "sidechain_or_pumping",
    ]
    for row in rows:
        scores = rhythm_scores(row)
        out = {
            "file": row["file"],
            "start": row["start"],
            "end": row["end"],
            "clip": row.get("clip", ""),
            "rhythm_candidates": "|".join(active(scores)),
            "audioset_top": row.get("audioset_top", ""),
            "project_candidates": row.get("top_labels", ""),
        }
        for label in labels:
            out[label] = scores[label]
            out[f"{label}_strength"] = strength(scores[label])
        out_rows.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        by_file[row["file"]].append(row)

    counter: Counter[str] = Counter()
    for row in out_rows:
        counter.update(row["rhythm_candidates"].split("|") if row["rhythm_candidates"] else [])

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        trs = []
        for row in file_rows:
            cells = "".join(f"<td>{bar(float(row[label]))}</td>" for label in labels)
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
                f"<td>{html.escape(row['rhythm_candidates'] or '-')}</td>"
                f"{cells}"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(file_name)}</h2>
<table>
<tr><th>Time</th><th>Audio</th><th>Rhythm Candidates</th>{''.join(f'<th>{label}</th>' for label in labels)}</tr>
{''.join(trs)}
</table>
</section>"""
        )

    summary = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Rhythm Section Detail</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
.bar {{ width: 86px; height: 9px; background: #eee; margin-bottom: 2px; }}
.bar span {{ display: block; height: 9px; background: #999; }}
.bar.strong span {{ background: #7b2d26; }}
.bar.medium span {{ background: #bd6658; }}
.bar.weak span {{ background: #d7aaa2; }}
small {{ color: #555; }}
</style>
<h1>Rhythm Section Detail</h1>
<p>This assumes mixed pop clips often contain multiple simultaneous layers. It asks what kind of rhythm/bass evidence is present, not whether drums exist at all.</p>
<h2>Summary</h2>
<table><tr><th>Candidate</th><th>Count</th></tr>{summary}</table>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/rhythm_section_detail.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/rhythm_section_detail.html"))
    args = parser.parse_args()
    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    write_outputs(rows, args.out_csv, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
