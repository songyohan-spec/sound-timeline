import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path


FEATURES = [
    "centroid",
    "bandwidth",
    "flatness",
    "zcr",
    "rolloff",
    "rms_std",
    "rms_range",
    "motion_strength",
    "motion_freq",
    "width",
]


def clamp01(value):
    return max(0.0, min(1.0, float(value)))


def read_features(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for key in FEATURES + ["start", "end"]:
                row[key] = float(row[key])
            row["segment_index"] = int(row["segment_index"])
            rows.append(row)
    return rows


def corpus_stats(rows):
    stats = {}
    for key in FEATURES:
        values = [row[key] for row in rows]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        stats[key] = (mean, var ** 0.5 + 1e-8)
    return stats


def z(row, key, stats):
    mean, std = stats[key]
    return (row[key] - mean) / std


def score_segment(row, stats):
    bright = clamp01((max(z(row, "centroid", stats), z(row, "rolloff", stats)) + 1.2) / 2.4)
    dark = 1.0 - bright
    noisy = clamp01((max(z(row, "flatness", stats), z(row, "zcr", stats)) + 1.2) / 2.4)
    clean = 1.0 - noisy
    wide = clamp01((z(row, "width", stats) + 1.2) / 2.4)
    narrow = 1.0 - wide
    moving = clamp01((max(z(row, "motion_strength", stats), z(row, "rms_range", stats), z(row, "rms_std", stats)) + 1.2) / 2.4)
    steady = 1.0 - moving
    low = clamp01((-min(z(row, "centroid", stats), z(row, "rolloff", stats)) + 0.8) / 2.2)
    transient = clamp01((z(row, "rms_range", stats) + z(row, "zcr", stats) + 1.8) / 3.6)

    return {
        "vocals": {
            "clean_vocal": 0.18 + 0.28 * clean + 0.20 * bright + 0.12 * narrow,
            "hard_tuned_vocal": 0.12 + 0.34 * bright + 0.20 * clean + 0.18 * moving,
            "pitched_vocal_chop": 0.14 + 0.34 * moving + 0.22 * bright + 0.10 * transient,
            "formant_shifted_vocal": 0.10 + 0.25 * noisy + 0.22 * moving + 0.12 * narrow,
            "breathy_vocal_pad": 0.10 + 0.28 * clean + 0.25 * wide + 0.12 * steady,
            "vocoder_vocal_texture": 0.10 + 0.30 * noisy + 0.20 * bright + 0.18 * steady,
            "stacked_harmony_vocal": 0.12 + 0.24 * wide + 0.24 * clean + 0.14 * steady,
        },
        "synth": {
            "airy_synth_pad": 0.12 + 0.26 * bright + 0.24 * wide + 0.18 * steady,
            "supersaw_pad": 0.10 + 0.26 * bright + 0.26 * wide + 0.16 * noisy,
            "brass_like_synth": 0.08 + 0.30 * bright + 0.20 * narrow + 0.16 * transient,
            "glassy_fm_bell": 0.08 + 0.36 * bright + 0.20 * clean + 0.16 * transient,
            "filtered_synth_pluck": 0.12 + 0.28 * dark + 0.22 * transient + 0.14 * narrow,
            "bitcrushed_synth_lead": 0.10 + 0.34 * bright + 0.34 * noisy + 0.12 * narrow,
            "noisy_wavetable_texture": 0.10 + 0.30 * noisy + 0.24 * wide + 0.14 * moving,
            "granular_synth_texture": 0.10 + 0.34 * noisy + 0.20 * wide + 0.22 * moving,
        },
        "bass": {
            "sub_bass": 0.14 + 0.46 * low + 0.18 * clean + 0.12 * narrow,
            "808_bass": 0.10 + 0.38 * low + 0.22 * transient + 0.16 * moving,
            "distorted_bass": 0.10 + 0.34 * low + 0.28 * noisy + 0.14 * moving,
            "reese_bass": 0.08 + 0.30 * low + 0.24 * wide + 0.18 * noisy,
            "pulsing_sidechain_bass": 0.10 + 0.34 * low + 0.34 * moving + 0.12 * dark,
            "synth_bass_pluck": 0.10 + 0.34 * low + 0.24 * transient + 0.12 * clean,
        },
        "drums": {
            "electronic_kick": 0.08 + 0.28 * low + 0.34 * transient + 0.12 * narrow,
            "acoustic_kick": 0.06 + 0.22 * low + 0.24 * transient + 0.14 * clean,
            "snare_clap": 0.08 + 0.22 * bright + 0.28 * transient + 0.14 * noisy,
            "rim_or_snap": 0.06 + 0.30 * bright + 0.30 * transient + 0.10 * narrow,
            "hi_hat_tick": 0.06 + 0.42 * bright + 0.24 * transient + 0.18 * noisy,
            "breakbeat_loop": 0.08 + 0.26 * transient + 0.26 * moving + 0.18 * noisy,
            "glitch_percussion": 0.08 + 0.28 * transient + 0.34 * noisy + 0.16 * bright,
            "tom_or_percussive_fill": 0.06 + 0.22 * low + 0.28 * transient + 0.14 * moving,
        },
        "guitar_strings": {
            "filtered_guitar_loop": 0.12 + 0.26 * dark + 0.22 * transient + 0.16 * clean,
            "washed_chorus_guitar": 0.10 + 0.28 * wide + 0.20 * clean + 0.14 * moving,
            "distorted_guitar_texture": 0.08 + 0.28 * noisy + 0.22 * bright + 0.16 * moving,
            "reverse_guitar_swell": 0.06 + 0.26 * moving + 0.18 * wide + 0.12 * clean,
            "string_pad": 0.10 + 0.26 * clean + 0.24 * wide + 0.18 * steady,
            "violin_like_lead": 0.08 + 0.32 * bright + 0.22 * clean + 0.12 * steady,
            "pizzicato_string_like": 0.06 + 0.26 * transient + 0.22 * clean + 0.14 * bright,
        },
        "noise_fx": {
            "vinyl_noise": 0.08 + 0.34 * noisy + 0.22 * steady + 0.10 * narrow,
            "tape_hiss": 0.08 + 0.36 * noisy + 0.18 * bright + 0.18 * steady,
            "riser_noise": 0.06 + 0.34 * noisy + 0.30 * moving + 0.14 * bright,
            "impact_tail": 0.06 + 0.28 * transient + 0.22 * wide + 0.16 * moving,
            "digital_glitch": 0.08 + 0.36 * noisy + 0.24 * transient + 0.18 * bright,
            "granular_texture": 0.10 + 0.38 * noisy + 0.26 * moving + 0.12 * wide,
            "reverse_swell": 0.06 + 0.28 * moving + 0.20 * wide + 0.12 * clean,
            "ambient_foley_texture": 0.08 + 0.22 * noisy + 0.22 * wide + 0.18 * steady,
        },
        "sampled_loop": {
            "filtered_sample_loop": 0.14 + 0.28 * dark + 0.20 * moving + 0.16 * transient,
            "resampled_pop_texture": 0.12 + 0.24 * noisy + 0.20 * moving + 0.18 * wide,
            "chopped_sample_loop": 0.12 + 0.30 * transient + 0.28 * moving + 0.12 * noisy,
            "vocal_synth_hybrid": 0.12 + 0.22 * bright + 0.22 * moving + 0.18 * clean,
            "unknown_hybrid_loop": 0.16 + 0.16 * moving + 0.16 * noisy + 0.12 * wide,
        },
        "treatments": {
            "dry_close": 0.18 + 0.34 * narrow + 0.28 * steady + 0.12 * clean,
            "long_reverb_wash": 0.10 + 0.36 * wide + 0.24 * steady + 0.14 * clean,
            "short_room": 0.12 + 0.20 * narrow + 0.20 * clean + 0.14 * transient,
            "delay_throw_tail": 0.08 + 0.22 * wide + 0.22 * moving + 0.12 * bright,
            "chorus_widened": 0.10 + 0.46 * wide + 0.12 * clean,
            "bitcrushed": 0.08 + 0.42 * noisy + 0.26 * bright,
            "saturated": 0.10 + 0.24 * noisy + 0.20 * dark + 0.12 * moving,
            "distorted": 0.08 + 0.34 * noisy + 0.18 * transient + 0.12 * bright,
            "filtered_dark": 0.12 + 0.48 * dark + 0.12 * steady,
            "filter_opening": 0.08 + 0.28 * moving + 0.18 * bright,
            "gated_chopped": 0.08 + 0.34 * transient + 0.30 * moving,
            "sidechain_pumping": 0.10 + 0.48 * moving + 0.16 * low,
            "tremolo_or_pulsing": 0.10 + 0.42 * moving + 0.10 * clean,
            "reversed": 0.06 + 0.24 * moving + 0.14 * wide,
            "pitch_warped": 0.06 + 0.24 * moving + 0.18 * noisy + 0.12 * bright,
        },
    }


def normalize_scores(scores):
    return {
        group: {
            label: round(clamp01(score), 4)
            for label, score in labels.items()
        }
        for group, labels in scores.items()
    }


def status(score, thresholds):
    if score >= thresholds["detected"]:
        return "detected"
    if score >= thresholds["possible"]:
        return "possible"
    return "weak"


def top_items(scores, thresholds, limit=4):
    items = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [
        {
            "label": label,
            "score": score,
            "status": status(score, thresholds),
        }
        for label, score in items
    ]


def summarize_clip(segment_profiles, thresholds):
    groups = [group for group in segment_profiles[0]["scores"] if group != "treatments"]
    summary = {"elements": {}, "treatments": []}
    for group in groups:
        totals = defaultdict(float)
        for profile in segment_profiles:
            for label, score in profile["scores"][group].items():
                totals[label] += score
        averaged = {label: score / len(segment_profiles) for label, score in totals.items()}
        summary["elements"][group] = top_items(averaged, thresholds, limit=3)

    totals = defaultdict(float)
    for profile in segment_profiles:
        for label, score in profile["scores"]["treatments"].items():
            totals[label] += score
    treatments = {label: score / len(segment_profiles) for label, score in totals.items()}
    summary["treatments"] = top_items(treatments, thresholds, limit=6)
    return summary


def build_profiles(rows, ontology):
    stats = corpus_stats(rows)
    thresholds = ontology["status_thresholds"]
    by_file = defaultdict(list)
    for row in sorted(rows, key=lambda item: (item["file"], item["start"])):
        scores = normalize_scores(score_segment(row, stats))
        profile = {
            "file": row["file"],
            "stem": row["stem"],
            "segment_index": row["segment_index"],
            "start": row["start"],
            "end": row["end"],
            "scores": scores,
            "top_by_group": {
                group: top_items(group_scores, thresholds, limit=4)
                for group, group_scores in scores.items()
            },
        }
        by_file[row["file"]].append(profile)
    return by_file


def write_json(by_file, ontology, out):
    data = {
        "mode": "slot_based_sound_element_profiler",
        "ontology": ontology,
        "files": [],
        "caution": "This first profiler is heuristic DSP scoring. It is a scaffold for element presence detection, not a trained source identifier yet.",
    }
    thresholds = ontology["status_thresholds"]
    for file, segments in by_file.items():
        data["files"].append(
            {
                "file": file,
                "summary": summarize_clip(segments, thresholds),
                "segments": segments,
            }
        )
    Path(out).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def item_text(items, include_weak=False):
    kept = [item for item in items if include_weak or item["status"] != "weak"]
    if not kept:
        kept = items[:2]
    return "<br>".join(
        f"{html.escape(item['label'])} <span class='{item['status']}'>({item['status']}, {item['score']:.2f})</span>"
        for item in kept
    )


def write_html(by_file, ontology, out):
    thresholds = ontology["status_thresholds"]
    sections = []
    for file, segments in by_file.items():
        summary = summarize_clip(segments, thresholds)
        summary_rows = []
        for group, items in summary["elements"].items():
            summary_rows.append(
                "<tr>"
                f"<th>{html.escape(group)}</th>"
                f"<td>{item_text(items)}</td>"
                "</tr>"
            )
        summary_rows.append(
            "<tr>"
            "<th>treatments</th>"
            f"<td>{item_text(summary['treatments'])}</td>"
            "</tr>"
        )

        segment_rows = []
        for segment in segments:
            audible_groups = []
            for group in ["vocals", "synth", "bass", "drums", "guitar_strings", "noise_fx", "sampled_loop"]:
                audible_groups.append(
                    f"<strong>{html.escape(group)}</strong><br>{item_text(segment['top_by_group'][group])}"
                )
            segment_rows.append(
                "<tr>"
                f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
                f"<td>{'<hr>'.join(audible_groups)}</td>"
                f"<td>{item_text(segment['top_by_group']['treatments'])}</td>"
                "</tr>"
            )

        sections.append(
            f"<section><h2>{html.escape(file)}</h2>"
            "<h3>Clip-Level Audible Elements</h3>"
            f"<table><tbody>{''.join(summary_rows)}</tbody></table>"
            "<h3>Segment Detail</h3>"
            "<table><thead><tr><th>Time</th><th>Element Slots</th><th>Treatment</th></tr></thead>"
            f"<tbody>{''.join(segment_rows)}</tbody></table></section>"
        )

    style = """
    body{font-family:Arial,sans-serif;max-width:1320px;margin:32px auto;color:#111}
    section{border-top:2px solid #111;margin-top:28px;padding-top:18px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 26px}
    th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}
    th{background:#eee;width:180px}
    .detected{font-weight:700;color:#111}
    .possible{color:#8a5a00}
    .weak{color:#777}
    hr{border:0;border-top:1px solid #eee;margin:8px 0}
    """
    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Sound Element Profile</title><style>{style}</style></head>
<body>
<h1>Sound Element Profile</h1>
<p>This view asks what audible elements are present in each clip. It is slot-based, so a segment can contain vocal, synth, bass, drums, guitar/strings, noise/fx, and sampled-loop cues at the same time.</p>
{''.join(sections)}
<p>Current scorer: heuristic DSP scaffold. Next step is replacing or augmenting scores with trained/semantic models.</p>
</body>
</html>
"""
    Path(out).write_text(doc, encoding="utf-8")


def write_csv(by_file, ontology, out):
    thresholds = ontology["status_thresholds"]
    fields = ["file", "start", "end", "group", "label", "score", "status"]
    with Path(out).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for file, segments in by_file.items():
            for segment in segments:
                for group, items in segment["top_by_group"].items():
                    for item in items:
                        if item["status"] == "weak":
                            continue
                        writer.writerow(
                            {
                                "file": file,
                                "start": f"{segment['start']:.2f}",
                                "end": f"{segment['end']:.2f}",
                                "group": group,
                                "label": item["label"],
                                "score": f"{item['score']:.4f}",
                                "status": status(item["score"], thresholds),
                            }
                        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--ontology", default="configs/sound_element_ontology.json")
    parser.add_argument("--filter-stem-prefix", default=None)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    ontology = json.loads(Path(args.ontology).read_text(encoding="utf-8"))
    rows = read_features(args.features)
    if args.filter_stem_prefix:
        rows = [row for row in rows if row["stem"].startswith(args.filter_stem_prefix)]
    by_file = build_profiles(rows, ontology)
    write_json(by_file, ontology, args.out_json)
    write_csv(by_file, ontology, args.out_csv)
    write_html(by_file, ontology, args.out_html)
    print(f"wrote: {args.out_json}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
