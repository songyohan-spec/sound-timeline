import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


SOURCE_HINTS = {
    "processed_lead_vocal": "dry rap/sung vocal phrases, then Auto-Tune/pitch correction, saturation, chorus, compression",
    "hard_tuned_vocal": "clean vocal phrases rendered through hard tuning or robotic pitch correction",
    "pitched_vocal_chop": "short vocal one-shots/chops, pitch shifted melodically, chopped/gated",
    "breathy_vocal_pad": "soft sustained vocal vowels, breathy backing vocals, washed pads",
    "stacked_harmony_vocal": "layered vocal harmonies, octave doubles, chorus stacks",
    "vocal_synth_hybrid": "vocal resynthesized, vocoder, formant/pitch hybrid textures",
    "vocoder_vocal_texture": "vocoder phrases, formant-shifted vocal slices",
    "lush_synth_pad": "wide sustained synth pads, supersaw/analog pad sources",
    "fuzzy_diy_synth_texture": "lo-fi/fuzzy synth loops, mild distortion, tape/bitcrush",
    "syrupy_video_game_synth_melody": "game-like lead loops, chiptune-ish melodic synths, bright arps",
    "rage_synth_lead": "rage/trap synth leads, saw leads, bright aggressive plucks",
    "bitcrushed_synth_lead": "bright synth lead through bitcrush/distortion",
    "glitching_bell_texture": "FM bells, mallets, music-box tones, glitchy bell loops",
    "synth_flute_or_recorder_like_lead": "sine whistle, recorder/flute-like synth lead, breathy pipe tones",
    "noisy_wavetable_texture": "wavetable noise textures, harsh digital synth beds",
    "granular_synth_texture": "granular pads, grain clouds, smeared chopped synth texture",
    "sub_bass": "clean sine/sub bass notes and loops",
    "distorted_808_bass": "808 bass one-shots/loops with drive, clipping, glide",
    "pulsing_sidechain_bass": "bass loops ducked with sidechain or volume-shaper pumping",
    "trap_hi_hat_rolls": "trap hats, rolls, ticks, fast subdivisions",
    "electronic_clap_snare": "electronic clap/snare one-shots and loops",
    "trap_drum_pattern": "trap drum loops with kick/snare/hat pattern",
    "glitch_percussion": "clicks, glitch hits, digital percussion, oscillator blips",
    "live_drum_layer": "live or alt-rock drum loops, room drums, cymbal wash",
    "washed_chorus_guitar": "chorus/reverb electric guitar loops, shoegaze guitars",
    "distorted_guitar_texture": "distorted guitar beds/riffs, grunge/industrial guitars",
    "feedback_haze": "guitar feedback, amp noise, sustained feedback layers",
    "filtered_guitar_loop": "muted/filtered guitar loops, plucked guitar samples",
    "string_pad": "sustained strings, synthetic string pads",
    "digital_glitch": "digital glitches, clicks, error sounds, crushed transitions",
    "granular_texture": "granular noise, chopped grains, smeared micro-samples",
    "industrial_noise_layer": "industrial noise beds, metallic noise, harsh loops",
    "watery_background_texture": "watery ambience, liquid foley, submerged texture",
    "vinyl_noise": "vinyl crackle/noise and lo-fi background beds",
    "filtered_sample_loop": "filtered loops from synth/guitar/vocal/samples",
    "chopped_sample_loop": "chopped/resliced sample loops",
    "resampled_pop_texture": "rendered/resampled pop phrases and processed loops",
    "unknown_hybrid_loop": "hybrid sample/synth/vocal loops that do not fit clean source classes",
    "wide_reverb_wash": "render any dry source through long/wide reverb",
    "sidechain_pumping": "render loops through sidechain or tremolo/volume pumping",
    "bitcrushed": "render synth/vocal/drum sources through bitcrusher",
    "distorted": "render synth/vocal/guitar/bass through distortion",
    "filtered_dark": "render sources through lowpass/dark filtering",
    "gated_chopped": "render sustained sources through gate/chopper rhythm",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def invert_label_map(label_map):
    alias_to_canonical = {}
    canonical_to_group = {}
    for group, labels in label_map["canonical_groups"].items():
        for canonical, aliases in labels.items():
            canonical_to_group[canonical] = group
            alias_to_canonical[canonical] = canonical
            for alias in aliases:
                alias_to_canonical[alias] = canonical
    return alias_to_canonical, canonical_to_group


def priority_rank(reference_targets, canonical):
    highest = set(reference_targets["training_priority_groups"].get("highest", []))
    medium = set(reference_targets["training_priority_groups"].get("medium", []))
    lower = set(reference_targets["training_priority_groups"].get("lower_for_now", []))
    if canonical in highest:
        return "highest"
    if canonical in medium:
        return "medium"
    if canonical in lower:
        return "lower"
    return "reference"


def build_rows(reference_targets, label_map):
    alias_to_canonical, canonical_to_group = invert_label_map(label_map)
    evidence = defaultdict(list)
    missing = defaultdict(list)

    for item in reference_targets["reference_set"]:
        ref = f"{item['artist']} - {item['track']}"
        for target in item["priority_targets"]:
            canonical = alias_to_canonical.get(target)
            if canonical:
                evidence[canonical].append(ref)
            else:
                missing[target].append(ref)

    rows = []
    for canonical, refs in sorted(evidence.items(), key=lambda pair: (-len(pair[1]), pair[0])):
        rows.append(
            {
                "canonical_label": canonical,
                "group": canonical_to_group.get(canonical, "unknown"),
                "priority": priority_rank(reference_targets, canonical),
                "reference_count": len(set(refs)),
                "references": " | ".join(sorted(set(refs))),
                "source_hint": SOURCE_HINTS.get(canonical, "collect dry examples and processed variations"),
                "target_min_examples": 250 if priority_rank(reference_targets, canonical) == "highest" else 120,
                "folder": f"data/reference_training_sources/{canonical}",
            }
        )

    missing_rows = [
        {
            "unmapped_target": target,
            "references": " | ".join(sorted(set(refs))),
        }
        for target, refs in sorted(missing.items())
    ]
    return rows, missing_rows


def write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_missing_csv(rows, path):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["unmapped_target", "references"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows, missing_rows, path):
    priority_order = {"highest": 0, "medium": 1, "reference": 2, "lower": 3}
    rows = sorted(rows, key=lambda row: (priority_order.get(row["priority"], 9), row["group"], row["canonical_label"]))
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(row['priority'])}</td>"
            f"<td>{html.escape(row['group'])}</td>"
            f"<td><strong>{html.escape(row['canonical_label'])}</strong></td>"
            f"<td>{row['reference_count']}</td>"
            f"<td>{html.escape(str(row['target_min_examples']))}</td>"
            f"<td>{html.escape(row['source_hint'])}</td>"
            f"<td>{html.escape(row['references'])}</td>"
            f"<td><code>{html.escape(row['folder'])}</code></td>"
            "</tr>"
        )

    missing_table = ""
    if missing_rows:
        missing_table = "<h2>Unmapped Targets</h2><table><thead><tr><th>Target</th><th>References</th></tr></thead><tbody>"
        missing_table += "".join(
            f"<tr><td>{html.escape(row['unmapped_target'])}</td><td>{html.escape(row['references'])}</td></tr>"
            for row in missing_rows
        )
        missing_table += "</tbody></table>"

    style = """
    body{font-family:Arial,sans-serif;max-width:1320px;margin:32px auto;color:#111}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 28px}
    th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}
    th{background:#eee}
    code{font-family:Consolas,monospace}
    """
    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Reference Training Plan</title><style>{style}</style></head>
<body>
<h1>Reference-Driven Training Plan</h1>
<p>This plan turns the current reference songs into concrete sound-element targets and sample collection folders.</p>
<table>
<thead><tr><th>Priority</th><th>Group</th><th>Canonical Label</th><th>Refs</th><th>Min Examples</th><th>What To Collect / Synthesize</th><th>Reference Evidence</th><th>Folder</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
{missing_table}
</body>
</html>
"""
    Path(path).write_text(doc, encoding="utf-8")


def create_dirs(rows):
    for row in rows:
        folder = Path(row["folder"])
        folder.mkdir(parents=True, exist_ok=True)
        readme = folder / "README.md"
        if not readme.exists():
            readme.write_text(
                "\n".join(
                    [
                        f"# {row['canonical_label']}",
                        "",
                        f"Group: {row['group']}",
                        f"Priority: {row['priority']}",
                        f"Target minimum examples: {row['target_min_examples']}",
                        "",
                        "Collect or synthesize:",
                        row["source_hint"],
                        "",
                        "Reference evidence:",
                        row["references"],
                        "",
                    ]
                ),
                encoding="utf-8",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-targets", default="configs/reference_sound_targets.json")
    parser.add_argument("--label-map", default="configs/reference_label_map.json")
    parser.add_argument("--out-csv", default="outputs/reference_training_plan.csv")
    parser.add_argument("--out-missing-csv", default="outputs/reference_training_unmapped.csv")
    parser.add_argument("--out-html", default="outputs/reference_training_plan.html")
    parser.add_argument("--create-dirs", action="store_true")
    args = parser.parse_args()

    reference_targets = load_json(args.reference_targets)
    label_map = load_json(args.label_map)
    rows, missing_rows = build_rows(reference_targets, label_map)
    write_csv(rows, args.out_csv)
    write_missing_csv(missing_rows, args.out_missing_csv)
    write_html(rows, missing_rows, args.out_html)
    if args.create_dirs:
        create_dirs(rows)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_missing_csv}")
    print(f"wrote: {args.out_html}")
    if args.create_dirs:
        print("created source folders under data/reference_training_sources")


if __name__ == "__main__":
    main()
