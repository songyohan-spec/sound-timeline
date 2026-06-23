from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_KINDS = [
    "clean_or_lead_vocal",
    "rap_or_spoken_vocal",
    "hard_tuned_vocal",
    "lead_or_hook_vocal",
    "pitched_vocal_chop",
    "formant_or_vocoder_vocal",
    "vocal_synth_hybrid",
    "vocal_pad_or_harmony",
    "warm_keys_or_organ",
    "synth_pad_or_wash",
    "supersaw_or_bright_synth_stack",
    "synth_pluck_or_bell",
    "arpeggio_or_sequence_synth",
    "digital_synth_lead",
    "bitcrushed_or_aliasing_synth",
    "fuzzy_distorted_synth",
    "wavetable_noise_synth",
    "granular_or_resampled_synth",
    "synth_bass",
    "sub_or_808_bass",
    "distorted_bass",
    "sidechained_bass_pulse",
    "ducked_mix_pulse",
    "electronic_drum_machine",
    "breakbeat_or_live_drums",
    "kick_or_low_hit",
    "snare_clap_or_snap",
    "hat_tick_or_click",
    "glitch_percussion",
    "clean_or_acoustic_guitar",
    "guitar_or_plucked_loop",
    "distorted_guitar_or_rock_texture",
    "washed_guitar_or_strings",
    "string_or_violin_like",
    "piano_or_keyboard_loop",
    "sampled_loop_texture",
    "chopped_or_stuttered_sample",
    "filtered_or_muffled_loop",
    "ambient_bed_or_foley",
    "noise_or_fx_transition",
    "riser_or_swell",
    "impact_or_tail",
]

ACTIVE_THRESHOLDS = {
    "noise_or_fx_transition": 0.14,
    "ambient_bed_or_foley": 0.12,
    "riser_or_swell": 0.11,
    "impact_or_tail": 0.11,
    "wavetable_noise_synth": 0.10,
    "fuzzy_distorted_synth": 0.10,
    "bitcrushed_or_aliasing_synth": 0.10,
    "digital_synth_lead": 0.10,
    "kick_or_low_hit": 0.10,
    "sampled_loop_texture": 0.095,
    "piano_or_keyboard_loop": 0.10,
}


def parse_score_list(value: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, score = part.rsplit(":", 1)
        try:
            out[label.strip()] = float(score.split("/", 1)[0].strip())
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


def tag_score(tags: dict[str, float], words: list[str]) -> float:
    total = 0.0
    for label, score in tags.items():
        lower = label.lower()
        if any(word in lower for word in words):
            total += score
    return total


def num(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except ValueError:
        return 0.0


def kind_scores(row: dict) -> dict[str, float]:
    project = parse_project_top(row.get("top_labels", ""))
    public = parse_score_list(row.get("public_sound_cue_scores", ""))
    tags = parse_score_list(row.get("audioset_top", ""))
    flatness = num(row, "flatness_noise")
    width = num(row, "width")
    motion = num(row, "motion_strength")
    centroid = num(row, "brightness_centroid")

    scores = {kind: 0.0 for kind in SOURCE_KINDS}

    scores["clean_or_lead_vocal"] = max(
        tag_score(tags, ["singing", "female singing", "male singing", "vocal music"]) * 0.95,
        project.get("processed_lead_vocal", 0.0) * 0.34,
    )
    scores["rap_or_spoken_vocal"] = max(
        tag_score(tags, ["speech", "rapping", "narration"]) * 0.95,
        public.get("spoken_or_processed_voice", 0.0) * 0.80,
    )
    scores["hard_tuned_vocal"] = max(
        project.get("hard_tuned_vocal", 0.0) * 0.72,
        project.get("processed_lead_vocal", 0.0) * 0.32 if centroid > 1200 else 0.0,
    )
    scores["lead_or_hook_vocal"] = max(
        public.get("lead_or_hook_vocal", 0.0),
        public.get("vocal_presence", 0.0) * 0.7,
        tag_score(tags, ["singing", "rapping", "vocal music", "female singing", "male singing"]),
        project.get("processed_lead_vocal", 0.0) * 0.42,
    )
    scores["pitched_vocal_chop"] = max(
        project.get("pitched_vocal_chop", 0.0) * 0.68,
        project.get("hard_tuned_vocal", 0.0) * 0.18,
        public.get("spoken_or_processed_voice", 0.0) * 0.28,
    )
    scores["formant_or_vocoder_vocal"] = max(
        project.get("vocoder_vocal_texture", 0.0) * 0.65,
        tag_score(tags, ["synthetic singing"]) * 0.9,
    )
    scores["vocal_synth_hybrid"] = max(
        project.get("vocal_synth_hybrid", 0.0) * 0.66,
        min(public.get("vocal_presence", 0.0), public.get("electronic_synth_texture", 0.0)) * 1.1,
    )
    scores["vocal_pad_or_harmony"] = max(
        project.get("stacked_harmony_vocal", 0.0) * 0.62,
        project.get("breathy_vocal_pad", 0.0) * 0.72,
        tag_score(tags, ["choir", "chorus"]) * 0.75,
        project.get("vocal_synth_hybrid", 0.0) * 0.32 if width > 0.60 else 0.0,
    )

    scores["warm_keys_or_organ"] = max(
        tag_score(tags, ["keyboard", "organ", "electric piano", "piano"]) * 0.72,
        project.get("unknown_hybrid_loop", 0.0) * 0.28,
    )
    scores["synth_pad_or_wash"] = max(
        project.get("lush_synth_pad", 0.0) * 0.70,
        public.get("electronic_synth_texture", 0.0) * (0.68 if width > 0.55 else 0.44),
        tag_score(tags, ["ambient", "synthesizer", "keyboard"]) * 0.70,
    )
    scores["supersaw_or_bright_synth_stack"] = max(
        project.get("lush_synth_pad", 0.0) * 0.42 if centroid > 1400 else 0.0,
        project.get("rage_synth_lead", 0.0) * 0.48,
        tag_score(tags, ["electronic music", "trance", "dance"]) * 0.35,
    )
    scores["synth_pluck_or_bell"] = max(
        public.get("bell_pluck_or_tiny_digital_hook", 0.0),
        project.get("glitching_bell_texture", 0.0) * 0.74,
        project.get("syrupy_video_game_synth_melody", 0.0) * 0.38,
        tag_score(tags, ["bell", "ringtone", "ding", "jingle", "tinkle", "ping", "chink", "clink"]),
    )
    scores["arpeggio_or_sequence_synth"] = max(
        project.get("syrupy_video_game_synth_melody", 0.0) * 0.50,
        tag_score(tags, ["ringtone", "jingle", "electronic music"]) * 0.25 if motion > 0.18 else 0.0,
    )
    scores["digital_synth_lead"] = max(
        project.get("bitcrushed_synth_lead", 0.0) * 0.72,
        project.get("rage_synth_lead", 0.0) * 0.72,
        project.get("syrupy_video_game_synth_melody", 0.0) * 0.60,
        public.get("electronic_synth_texture", 0.0) * 0.25 if centroid > 1200 else 0.0,
    )
    scores["bitcrushed_or_aliasing_synth"] = max(
        project.get("bitcrushed_synth_lead", 0.0) * 0.58,
        project.get("fuzzy_diy_synth_texture", 0.0) * 0.26,
        tag_score(tags, ["chip", "8-bit", "distortion"]) * 0.35,
    )
    scores["fuzzy_distorted_synth"] = max(
        project.get("fuzzy_diy_synth_texture", 0.0) * 0.68,
        project.get("noisy_wavetable_texture", 0.0) * 0.30 if flatness > 0.16 else 0.0,
        tag_score(tags, ["distortion", "fuzz"]) * 0.40,
    )
    scores["wavetable_noise_synth"] = max(
        project.get("noisy_wavetable_texture", 0.0) * (0.62 if flatness > 0.14 else 0.34),
        public.get("electronic_synth_texture", 0.0) * 0.20 if flatness > 0.18 else 0.0,
    )
    scores["granular_or_resampled_synth"] = max(
        project.get("granular_synth_texture", 0.0) * 0.80,
        project.get("noisy_wavetable_texture", 0.0) * 0.32 if flatness > 0.20 else 0.0,
        project.get("resampled_pop_texture", 0.0) * 0.36,
    )

    scores["synth_bass"] = max(
        project.get("pulsing_sidechain_bass", 0.0) * 0.44,
        tag_score(tags, ["synth bass", "bass"]) * 0.38,
        public.get("club_or_bass_music_influence", 0.0) * 0.40,
    )
    scores["sub_or_808_bass"] = max(
        project.get("sub_bass", 0.0) * 0.72,
        project.get("distorted_808_bass", 0.0) * 0.60,
        tag_score(tags, ["bass", "dubstep", "drum and bass"]) * 0.45,
    )
    scores["distorted_bass"] = max(
        project.get("distorted_808_bass", 0.0) * 0.72,
        tag_score(tags, ["distortion", "dubstep"]) * 0.35,
    )
    scores["sidechained_bass_pulse"] = max(
        project.get("pulsing_sidechain_bass", 0.0) * 0.66,
        public.get("club_or_bass_music_influence", 0.0) * 0.70,
        min(0.18, motion * 0.55) if motion > 0.22 else 0.0,
    )
    scores["ducked_mix_pulse"] = max(
        min(0.20, motion * 0.62) if motion > 0.24 else 0.0,
        project.get("pulsing_sidechain_bass", 0.0) * 0.34,
    )
    scores["electronic_drum_machine"] = max(
        project.get("trap_drum_pattern", 0.0) * 0.62,
        public.get("drum_or_percussion_presence", 0.0) * 0.55,
        tag_score(tags, ["drum machine", "beat", "electronic music", "techno"]) * 0.40,
    )
    scores["breakbeat_or_live_drums"] = max(
        project.get("live_drum_layer", 0.0) * 0.72,
        public.get("drum_or_percussion_presence", 0.0) * 0.40,
        tag_score(tags, ["breakbeat", "drum kit", "drum roll", "rock", "percussion"]) * 0.70,
    )
    scores["kick_or_low_hit"] = max(
        project.get("trap_drum_pattern", 0.0) * 0.38,
        tag_score(tags, ["bass drum", "kick drum", "thump", "bang"]) * 0.9,
    )
    scores["snare_clap_or_snap"] = max(
        project.get("electronic_clap_snare", 0.0) * 0.70,
        tag_score(tags, ["snare", "clap", "snap"]) * 0.9,
    )
    scores["hat_tick_or_click"] = max(
        project.get("trap_hi_hat_rolls", 0.0) * 0.72,
        tag_score(tags, ["hi-hat", "tick", "click", "chink"]) * 0.75,
    )
    scores["glitch_percussion"] = max(
        project.get("glitch_percussion", 0.0) * (0.62 if flatness > 0.14 else 0.34),
        tag_score(tags, ["click", "mechanical", "drum machine"]) * 0.35 if flatness > 0.18 else 0.0,
    )

    scores["clean_or_acoustic_guitar"] = max(
        tag_score(tags, ["acoustic guitar", "strum"]) * 0.9,
        project.get("filtered_guitar_loop", 0.0) * 0.28,
    )
    scores["guitar_or_plucked_loop"] = max(
        public.get("guitar_or_plucked_string", 0.0),
        project.get("filtered_guitar_loop", 0.0) * 0.70,
        tag_score(tags, ["guitar", "strum", "plucked string"]) * 0.9,
    )
    scores["distorted_guitar_or_rock_texture"] = max(
        project.get("distorted_guitar_texture", 0.0) * 0.72,
        public.get("rock_guitar_energy", 0.0) * 0.80,
        tag_score(tags, ["electric guitar", "distortion", "rock", "grunge", "metal"]) * 0.55,
    )
    scores["washed_guitar_or_strings"] = max(
        project.get("washed_chorus_guitar", 0.0) * 0.70,
        project.get("string_pad", 0.0) * 0.72,
        project.get("feedback_haze", 0.0) * 0.60,
        tag_score(tags, ["string", "violin", "guitar"]) * 0.35 if width > 0.6 else 0.0,
    )
    scores["string_or_violin_like"] = max(
        project.get("string_pad", 0.0) * 0.72,
        tag_score(tags, ["violin", "string", "cello"]) * 0.85,
    )
    scores["piano_or_keyboard_loop"] = max(
        tag_score(tags, ["piano", "keyboard", "electric piano"]) * 0.78,
        project.get("unknown_hybrid_loop", 0.0) * 0.25,
    )

    scores["sampled_loop_texture"] = max(
        public.get("sampled_or_resampled_loop", 0.0),
        project.get("filtered_sample_loop", 0.0) * 0.70,
        project.get("unknown_hybrid_loop", 0.0) * 0.55,
        project.get("resampled_pop_texture", 0.0) * 0.75,
        public.get("electronic_synth_texture", 0.0) * 0.26 if motion > 0.16 else 0.0,
    )
    scores["chopped_or_stuttered_sample"] = max(
        project.get("chopped_sample_loop", 0.0) * 0.72,
        project.get("pitched_vocal_chop", 0.0) * 0.22,
        project.get("glitch_percussion", 0.0) * 0.22,
    )
    scores["filtered_or_muffled_loop"] = max(
        project.get("filtered_sample_loop", 0.0) * 0.50,
        project.get("filtered_guitar_loop", 0.0) * 0.42,
        project.get("unknown_hybrid_loop", 0.0) * 0.32 if centroid < 1100 else 0.0,
    )
    scores["ambient_bed_or_foley"] = max(
        project.get("watery_background_texture", 0.0) * 0.72,
        project.get("industrial_noise_layer", 0.0) * 0.34,
        tag_score(tags, ["ambient", "background", "inside", "outside", "vehicle", "water"]) * 0.45,
    )
    scores["noise_or_fx_transition"] = max(
        public.get("hit_or_fx_transient", 0.0),
        public.get("noise_bed_or_artifact", 0.0) * 0.72,
        project.get("digital_glitch", 0.0) * 0.70,
        project.get("industrial_noise_layer", 0.0) * 0.72,
        tag_score(tags, ["whoosh", "burst", "explosion"]) * 0.8,
        tag_score(tags, ["static", "noise"]) * 0.45 if flatness > 0.22 else 0.0,
        min(0.14, flatness * 0.45) if flatness > 0.28 else 0.0,
    )
    scores["riser_or_swell"] = max(
        project.get("feedback_haze", 0.0) * 0.32 if motion > 0.18 else 0.0,
        tag_score(tags, ["whoosh", "swoosh", "wind"]) * 0.70,
        min(0.14, motion * 0.45) if motion > 0.26 and width > 0.65 else 0.0,
    )
    scores["impact_or_tail"] = max(
        public.get("hit_or_fx_transient", 0.0) * 0.80,
        tag_score(tags, ["bang", "burst", "explosion", "thump", "impact"]) * 0.85,
    )

    return {key: round(value, 5) for key, value in scores.items()}


def evidence_basis(row: dict, label: str) -> str:
    project = parse_project_top(row.get("top_labels", ""))
    public = parse_score_list(row.get("public_sound_cue_scores", ""))
    tags = parse_score_list(row.get("audioset_top", ""))
    flatness = num(row, "flatness_noise")
    width = num(row, "width")
    motion = num(row, "motion_strength")

    basis: list[str] = []

    if label in {
        "clean_or_lead_vocal",
        "rap_or_spoken_vocal",
        "lead_or_hook_vocal",
        "warm_keys_or_organ",
        "synth_pad_or_wash",
        "synth_pluck_or_bell",
        "synth_bass",
        "sub_or_808_bass",
        "distorted_bass",
        "sidechained_bass_pulse",
        "ducked_mix_pulse",
        "electronic_drum_machine",
        "breakbeat_or_live_drums",
        "kick_or_low_hit",
        "snare_clap_or_snap",
        "hat_tick_or_click",
        "glitch_percussion",
        "clean_or_acoustic_guitar",
        "guitar_or_plucked_loop",
        "distorted_guitar_or_rock_texture",
        "washed_guitar_or_strings",
        "string_or_violin_like",
        "piano_or_keyboard_loop",
        "sampled_loop_texture",
        "ambient_bed_or_foley",
        "noise_or_fx_transition",
        "riser_or_swell",
        "impact_or_tail",
    }:
        if public or tags:
            basis.append("external_ast")
    if label in {
        "hard_tuned_vocal",
        "pitched_vocal_chop",
        "formant_or_vocoder_vocal",
        "vocal_synth_hybrid",
        "vocal_pad_or_harmony",
        "warm_keys_or_organ",
        "synth_pad_or_wash",
        "supersaw_or_bright_synth_stack",
        "synth_pluck_or_bell",
        "arpeggio_or_sequence_synth",
        "digital_synth_lead",
        "bitcrushed_or_aliasing_synth",
        "fuzzy_distorted_synth",
        "wavetable_noise_synth",
        "granular_or_resampled_synth",
        "synth_bass",
        "sub_or_808_bass",
        "distorted_bass",
        "sidechained_bass_pulse",
        "ducked_mix_pulse",
        "electronic_drum_machine",
        "breakbeat_or_live_drums",
        "kick_or_low_hit",
        "snare_clap_or_snap",
        "hat_tick_or_click",
        "glitch_percussion",
        "clean_or_acoustic_guitar",
        "guitar_or_plucked_loop",
        "distorted_guitar_or_rock_texture",
        "washed_guitar_or_strings",
        "string_or_violin_like",
        "piano_or_keyboard_loop",
        "sampled_loop_texture",
        "chopped_or_stuttered_sample",
        "filtered_or_muffled_loop",
        "ambient_bed_or_foley",
        "noise_or_fx_transition",
        "riser_or_swell",
        "impact_or_tail",
    }:
        if project:
            basis.append("project_model")
    if label in {
        "synth_pad_or_wash",
        "supersaw_or_bright_synth_stack",
        "arpeggio_or_sequence_synth",
        "bitcrushed_or_aliasing_synth",
        "fuzzy_distorted_synth",
        "wavetable_noise_synth",
        "granular_or_resampled_synth",
        "sidechained_bass_pulse",
        "ducked_mix_pulse",
        "electronic_drum_machine",
        "glitch_percussion",
        "washed_guitar_or_strings",
        "sampled_loop_texture",
        "chopped_or_stuttered_sample",
        "filtered_or_muffled_loop",
        "ambient_bed_or_foley",
        "noise_or_fx_transition",
        "riser_or_swell",
    }:
        if width > 0.55 or flatness > 0.14 or motion > 0.16:
            basis.append("dsp_hint")

    return "+".join(dict.fromkeys(basis)) or "weak_inference"


def strength(score: float) -> str:
    if score >= 0.18:
        return "strong"
    if score >= 0.10:
        return "medium"
    if score >= 0.06:
        return "weak"
    return ""


def active(scores: dict[str, float], limit: int = 6) -> list[str]:
    return [
        label
        for label, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if score >= ACTIVE_THRESHOLDS.get(label, 0.085)
    ][:limit]


def bar(score: float) -> str:
    width = min(100, int(score * 300))
    cls = strength(score) or "none"
    return f"<div class='bar {cls}'><span style='width:{width}%'></span></div><small>{score:.3f}</small>"


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    out_rows = []
    counter: Counter[str] = Counter()
    primary_counter: Counter[str] = Counter()
    for row in rows:
        scores = kind_scores(row)
        active_kinds = active(scores)
        primary = active_kinds[0] if active_kinds else "unclear"
        counter.update(active_kinds)
        primary_counter[primary] += 1
        out = {
            "file": row["file"],
            "start": row["start"],
            "end": row["end"],
            "clip": row.get("clip", ""),
            "primary_source_kind": primary,
            "active_source_kinds": "|".join(active_kinds),
            "active_source_kind_evidence": "|".join(f"{label}:{evidence_basis(row, label)}" for label in active_kinds),
        }
        out.update(scores)
        out_rows.append(out)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in out_rows:
        by_file[row["file"]].append(row)

    def count_table(title: str, values: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in values.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Source Kind</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for file_name, file_rows in sorted(by_file.items()):
        trs = []
        for row in file_rows:
            score_lines = "<br>".join(
                f"{label}: {bar(float(row[label]))}"
                for label in SOURCE_KINDS
                if float(row[label]) >= 0.065
            ) or "-"
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(row['clip'])}'></audio></td>"
                f"<td>{html.escape(row['primary_source_kind'])}</td>"
                f"<td>{html.escape(row['active_source_kinds'] or '-')}<br><small>{html.escape(row.get('active_source_kind_evidence', '') or '-')}</small></td>"
                f"<td>{score_lines}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(file_name)}</h2>
<table>
<tr><th>Time</th><th>Audio</th><th>Primary Source Kind</th><th>Active Source Kinds / Evidence</th><th>Scores</th></tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Source Kind Detail</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 26px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 190px; }}
.bar {{ display:inline-block; width:80px; height:8px; background:#eee; margin:0 4px; }}
.bar span {{ display:block; height:8px; background:#999; }}
.bar.strong span {{ background:#265a9b; }}
.bar.medium span {{ background:#6b9fcf; }}
.bar.weak span {{ background:#b7cbe0; }}
small {{ color:#555; }}
</style>
<h1>Source Kind Detail</h1>
<p>This report expands the source vocabulary at a middle level: not exact stems or presets, but more useful source-kind hypotheses than broad vocals/synth/drums labels. Evidence tags show whether a candidate is mainly supported by the public AST/AudioSet panel, the project model, or DSP hints.</p>
{count_table("Primary Source Kind", primary_counter)}
{count_table("Active Source Kind Mentions", counter)}
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/audioset_sound_cues.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.html"))
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    write_outputs(rows, args.out_csv, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
