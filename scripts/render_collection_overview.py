from __future__ import annotations

import argparse
import csv
import html
import re
from collections import Counter, defaultdict
from pathlib import Path


LAYER_COLS = ["vocals", "synth", "guitar_strings", "bass", "drums", "noise_fx", "sampled_loop"]
VOCAL_COLS = [
    "vocal_lead_or_hook_vocal",
    "vocal_spoken_processed_vocal",
    "vocal_hard_tuned_vocal",
    "vocal_pitched_vocal_chop",
    "vocal_breathy_vocal_pad",
    "vocal_stacked_harmony",
    "vocal_vocal_synth_hybrid",
    "vocal_vocoder_or_synthetic_vocal",
]
SYNTH_COLS = [
    "synth_synth_pad_or_wash",
    "synth_digital_pluck_or_bell",
    "synth_bitcrushed_synth_lead",
    "synth_noisy_wavetable_texture",
    "synth_game_like_synth_melody",
    "synth_filtered_sample_or_synth_loop",
    "synth_ambient_electronic_texture",
    "synth_bass_synth_pulse",
]
RHYTHM_COLS = [
    "kick_or_low_hit",
    "808_or_sub_bass",
    "snare_or_clap",
    "hi_hat_or_tick",
    "glitch_percussion",
    "breakbeat_or_drum_loop",
    "sidechain_or_pumping",
]


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def group_name(file_name: str) -> str:
    stem = Path(file_name).stem.strip()
    stem = re.sub(r"\s+", "", stem)
    return re.sub(r"[_-]?\d+$", "", stem)


def key(row: dict) -> tuple[str, str, str]:
    return (row["file"], row["start"], row["end"])


def f(row: dict, col: str) -> float:
    try:
        return float(row.get(col, 0.0) or 0.0)
    except ValueError:
        return 0.0


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def mean_scores(rows: list[dict], cols: list[str]) -> dict[str, float]:
    if not rows:
        return {col: 0.0 for col in cols}
    return {col: sum(f(row, col) for row in rows) / len(rows) for col in cols}


def top_mean(rows: list[dict], cols: list[str], limit: int = 5) -> list[tuple[str, float]]:
    means = mean_scores(rows, cols)
    return sorted(means.items(), key=lambda item: item[1], reverse=True)[:limit]


def top_counter(rows: list[dict], col: str, limit: int = 6) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(split_pipe(row.get(col, "")))
    return counter.most_common(limit)


def fmt_items(items: list[tuple[str, float | int]]) -> str:
    if not items:
        return "-"
    out = []
    for label, value in items:
        if isinstance(value, float):
            out.append(f"{html.escape(label)} <span>{value:.3f}</span>")
        else:
            out.append(f"{html.escape(label)} <span>{value}</span>")
    return "<br>".join(out)


def write_html(layer_rows: list[dict], vocal_rows: list[dict], rhythm_rows: list[dict], timeline_rows: list[dict], out_html: Path) -> None:
    vocal_by_key = {key(row): row for row in vocal_rows}
    rhythm_by_key = {key(row): row for row in rhythm_rows}
    timeline_by_key = {key(row): row for row in timeline_rows}

    groups: dict[str, list[dict]] = defaultdict(list)
    merged_rows = []
    for layer in layer_rows:
        merged = dict(layer)
        merged.update({f"vocal_detail_{k}": v for k, v in vocal_by_key.get(key(layer), {}).items()})
        merged.update({f"rhythm_detail_{k}": v for k, v in rhythm_by_key.get(key(layer), {}).items()})
        merged.update({f"timeline_{k}": v for k, v in timeline_by_key.get(key(layer), {}).items()})
        merged_rows.append(merged)
        groups[group_name(layer["file"])].append(merged)

    summary_rows = []
    for name, rows in sorted(groups.items()):
        layer_top = top_mean(rows, LAYER_COLS, 4)
        vocal_top = top_mean([{k.replace("vocal_detail_", ""): v for k, v in row.items() if k.startswith("vocal_detail_")} for row in rows], VOCAL_COLS, 4)
        synth_top = top_mean([{k.replace("vocal_detail_", ""): v for k, v in row.items() if k.startswith("vocal_detail_")} for row in rows], SYNTH_COLS, 4)
        rhythm_top = top_mean([{k.replace("rhythm_detail_", ""): v for k, v in row.items() if k.startswith("rhythm_detail_")} for row in rows], RHYTHM_COLS, 4)
        primary_top = top_counter([{ "primary": row.get("timeline_primary_read", "") } for row in rows], "primary", 4)
        summary_rows.append(
            "<tr>"
            f"<td><b>{html.escape(name)}</b><br><span>{len(rows)} segments</span></td>"
            f"<td>{fmt_items(layer_top)}</td>"
            f"<td>{fmt_items(primary_top)}</td>"
            f"<td>{fmt_items(vocal_top)}</td>"
            f"<td>{fmt_items(synth_top)}</td>"
            f"<td>{fmt_items(rhythm_top)}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Collection Overview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 8px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
span {{ color: #666; font-size: 12px; }}
</style>
<h1>Collection Overview</h1>
<p>Grouped by filename prefix. Use this to see which songs/clips are vocal-heavy, synth-heavy, rhythm-heavy, or texture-heavy.</p>
<table>
<tr><th>Group</th><th>Layer Mean</th><th>Primary Reads</th><th>Vocal Detail Mean</th><th>Synth Detail Mean</th><th>Rhythm Detail Mean</th></tr>
{''.join(summary_rows)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_layer_matrix.csv"))
    parser.add_argument("--vocal-synth", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/vocal_synth_detail.csv"))
    parser.add_argument("--rhythm", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/rhythm_section_detail.csv"))
    parser.add_argument("--timeline", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/sound_element_timeline.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/collection_overview.html"))
    args = parser.parse_args()
    write_html(load_csv(args.layer), load_csv(args.vocal_synth), load_csv(args.rhythm), load_csv(args.timeline), args.out_html)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
