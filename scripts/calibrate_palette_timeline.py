from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_GROUPS = {
    "filtered_sample_loop": {
        "filtered_sample_loop",
        "processed_melodic_loop",
        "resampled_pop_texture",
        "chopped_melodic_texture",
        "warped_pluck_loop",
    },
    "vocal_synth_chop": {
        "vocal_synth_hybrid",
        "pitched_vocal_chop",
        "formant_shifted_vocal_chop",
        "hard_tuned_vocal_lead",
        "vocoder_vocal_texture",
        "layered_harmony_stack",
    },
    "reverse_or_swell_texture": {
        "reverse_vocal_swell",
        "reverse_guitar_swell",
    },
    "processed_guitar_loop": {
        "filtered_guitar_loop",
        "chorus_arpeggio_loop",
        "pitch_warped_guitar",
        "chorus_guitar_wash",
        "distorted_guitar_texture",
    },
    "synth_lead_or_pluck": {
        "bitcrushed_synth_lead",
        "filtered_synth_pluck",
        "glassy_fm_bell",
        "arpeggiated_synth_sequence",
        "analog_bass_pulse",
    },
}

PROCESSING_GROUPS = {
    "pulsed_or_ducked_motion": {
        "sidechain_pumping",
        "rhythmic_pulse_motion",
        "gated_stutter_motion",
        "chopped_retriggered_envelope",
    },
    "delay_or_echo_tail": {
        "delay_throw_tail",
    },
    "dry_or_close": {
        "dry_close",
    },
    "warbly_tape_motion": {
        "tape_wow_flutter",
    },
    "modulated_wide_space": {
        "wide_chorus_widening",
        "phaser_flanger_motion",
    },
    "long_reverb_space": {
        "long_reverb_wash",
    },
}


def group_for(label: str, groups: dict[str, set[str]]) -> str:
    for group, labels in groups.items():
        if label in labels:
            return group
    return label or "none"


def score_groups(items: list[dict], groups: dict[str, set[str]]) -> tuple[str, float, list[str]]:
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    for item in items:
        label = str(item.get("label") or "")
        if not label:
            continue
        group = group_for(label, groups)
        score = float(item.get("score") or 0.0)
        scores[group] += score
        evidence[group].append(label)
    if not scores:
        return "none", 0.0, []
    group, score = max(scores.items(), key=lambda pair: pair[1])
    return group, score, evidence[group]


def build_regions(segments: list[dict]) -> list[dict]:
    regions = []
    for segment in segments:
        key = (segment["calibrated_source"], segment["calibrated_processing"])
        if regions and regions[-1]["key"] == key and abs(regions[-1]["end"] - segment["start"]) < 0.001:
            regions[-1]["end"] = segment["end"]
            regions[-1]["segments"].append(segment["index"])
            continue
        regions.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "key": key,
                "calibrated_source": segment["calibrated_source"],
                "source_score": segment["source_score"],
                "source_evidence": segment["source_evidence"],
                "calibrated_processing": segment["calibrated_processing"],
                "processing_score": segment["processing_score"],
                "processing_evidence": segment["processing_evidence"],
                "segments": [segment["index"]],
            }
        )
    for region in regions:
        region.pop("key", None)
    return regions


def render_html(report: dict) -> str:
    region_rows = []
    for region in report["regions"]:
        region_rows.append(
            "<tr>"
            f"<td>{region['start']:.2f}-{region['end']:.2f}s</td>"
            f"<td>{region['calibrated_source']}</td>"
            f"<td>{region['source_score']:.3f}</td>"
            f"<td>{', '.join(region['source_evidence'][:4])}</td>"
            f"<td>{region['calibrated_processing']}</td>"
            f"<td>{region['processing_score']:.3f}</td>"
            f"<td>{', '.join(region['processing_evidence'][:4])}</td>"
            "</tr>"
        )

    segment_rows = []
    for segment in report["segments"]:
        segment_rows.append(
            "<tr>"
            f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
            f"<td>{segment['calibrated_source']}</td>"
            f"<td>{segment['source_score']:.3f}</td>"
            f"<td>{', '.join(segment['source_evidence'][:5])}</td>"
            f"<td>{segment['calibrated_processing']}</td>"
            f"<td>{segment['processing_score']:.3f}</td>"
            f"<td>{', '.join(segment['processing_evidence'][:5])}</td>"
            "</tr>"
        )

    return "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            f"<title>{report['title']} Calibrated Timeline</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 28px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.note{color:#555}</style>",
            f"<h1>{report['title']} Calibrated Sound Timeline</h1>",
            "<p class='note'>Macro labels group unstable fine labels into safer sound-family hypotheses.</p>",
            "<h2>Region Summary</h2>",
            "<table><thead><tr><th>Time</th><th>Calibrated Source</th><th>Source Score</th><th>Source Evidence</th><th>Calibrated Processing</th><th>Processing Score</th><th>Processing Evidence</th></tr></thead><tbody>",
            *region_rows,
            "</tbody></table>",
            "<h2>Segment Detail</h2>",
            "<table><thead><tr><th>Time</th><th>Calibrated Source</th><th>Source Score</th><th>Source Evidence</th><th>Calibrated Processing</th><th>Processing Score</th><th>Processing Evidence</th></tr></thead><tbody>",
            *segment_rows,
            "</tbody></table>",
        ]
    )


def calibrate(timeline: dict) -> dict:
    segments = []
    for segment in timeline.get("segments", []):
        profile = segment.get("profile", {})
        source, source_score, source_evidence = score_groups(profile.get("source_layers", []), SOURCE_GROUPS)
        processing, processing_score, processing_evidence = score_groups(profile.get("processing_cues", []), PROCESSING_GROUPS)
        segments.append(
            {
                "index": segment.get("index"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "calibrated_source": source,
                "source_score": round(source_score, 6),
                "source_evidence": source_evidence,
                "calibrated_processing": processing,
                "processing_score": round(processing_score, 6),
                "processing_evidence": processing_evidence,
            }
        )
    return {
        "title": timeline.get("title", "Timeline"),
        "audio": timeline.get("audio"),
        "mode": "calibrated_macro_sound_timeline",
        "regions": build_regions(segments),
        "segments": segments,
        "source_group_counts": dict(Counter(segment["calibrated_source"] for segment in segments).most_common()),
        "processing_group_counts": dict(Counter(segment["calibrated_processing"] for segment in segments).most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    timeline = json.loads(args.input.read_text(encoding="utf-8"))
    report = calibrate(timeline)
    out_json = args.out_json or args.input.with_name(args.input.stem.replace("_palette_timeline", "_calibrated_timeline") + ".json")
    out_html = args.out_html or args.input.with_name(args.input.stem.replace("_palette_timeline", "_calibrated_timeline") + ".html")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_html.write_text(render_html(report), encoding="utf-8")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()
