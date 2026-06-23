from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


LAYER_ORDER = ["vocals", "drums", "bass", "synth", "guitar_keys", "sample_fx", "noise_fx"]
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


LABEL_TO_LAYER = {
    "clean_or_lead_vocal": "vocals",
    "lead_or_hook_vocal": "vocals",
    "rap_or_spoken_vocal": "vocals",
    "hard_tuned_vocal": "vocals",
    "pitched_vocal_chop": "vocals",
    "vocal_pad_or_harmony": "vocals",
    "formant_or_vocoder_vocal": "vocals",
    "vocal_synth_hybrid": "vocals",
    "electronic_drum_machine": "drums",
    "breakbeat_or_live_drums": "drums",
    "kick_or_low_hit": "drums",
    "snare_clap_or_snap": "drums",
    "hat_tick_or_click": "drums",
    "glitch_percussion": "drums",
    "synth_bass": "bass",
    "sub_or_808_bass": "bass",
    "distorted_bass": "bass",
    "sidechained_bass_pulse": "bass",
    "synth_pad_or_wash": "synth",
    "supersaw_or_bright_synth_stack": "synth",
    "synth_pluck_or_bell": "synth",
    "arpeggio_or_sequence_synth": "synth",
    "digital_synth_lead": "synth",
    "bitcrushed_or_aliasing_synth": "synth",
    "fuzzy_distorted_synth": "synth",
    "wavetable_noise_synth": "synth",
    "granular_or_resampled_synth": "synth",
    "guitar_or_plucked_loop": "guitar_keys",
    "clean_or_acoustic_guitar": "guitar_keys",
    "distorted_guitar_or_rock_texture": "guitar_keys",
    "washed_guitar_or_strings": "guitar_keys",
    "string_or_violin_like": "guitar_keys",
    "piano_or_keyboard_loop": "guitar_keys",
    "warm_keys_or_organ": "guitar_keys",
    "sampled_loop_texture": "sample_fx",
    "chopped_or_stuttered_sample": "sample_fx",
    "filtered_or_muffled_loop": "sample_fx",
    "ambient_bed_or_foley": "noise_fx",
    "noise_or_fx_transition": "noise_fx",
    "riser_or_swell": "noise_fx",
    "impact_or_tail": "noise_fx",
}


SYNTH_DETAIL_TO_SOURCE_KIND = {
    "synth_pad_wash": "synth_pad_or_wash",
    "supersaw_stack": "supersaw_or_bright_synth_stack",
    "digital_synth_lead": "digital_synth_lead",
    "bitcrushed_synth_lead": "bitcrushed_or_aliasing_synth",
    "synth_pluck_bell": "synth_pluck_or_bell",
    "arpeggio_sequence": "arpeggio_or_sequence_synth",
    "granular_texture": "granular_or_resampled_synth",
    "wavetable_noise": "wavetable_noise_synth",
    "fuzzy_lofi_synth": "fuzzy_distorted_synth",
    "vocal_synth_hybrid": "vocal_synth_hybrid",
    "formant_vocoder": "formant_or_vocoder_vocal",
    "synth_bass": "synth_bass",
    "sidechained_synth_bass": "sidechained_bass_pulse",
    "sub_808_synth_bass": "sub_or_808_bass",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def norm_time(value: str) -> str:
    return str(float(value))


def key(row: dict) -> tuple[str, str, str, str]:
    return row["track"], row["stem"], norm_time(row["start"]), norm_time(row["end"])


def time_key(row: dict) -> tuple[str, str, str]:
    return row["track"], norm_time(row["start"]), norm_time(row["end"])


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_top_source_kinds(value: str) -> list[dict]:
    out = []
    for raw in str(value or "").split(";"):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        label, rest = raw.split(":", 1)
        parts = rest.split("/")
        try:
            score = float(parts[0])
            threshold = float(parts[1]) if len(parts) > 1 else 0.0
        except ValueError:
            continue
        status = parts[2] if len(parts) > 2 else "possible"
        ratio = score / threshold if threshold > 0 else score
        out.append({"label": label.strip(), "score": score, "threshold": threshold, "status": status, "ratio": ratio})
    return out


def candidate_strength(item: dict) -> str:
    if item["status"] == "detected" and item["ratio"] >= 1.35:
        return "strong"
    if item["status"] == "detected":
        return "likely"
    if item["ratio"] >= 0.92:
        return "possible"
    return ""


def strength_rank(value: str) -> int:
    return {"strong": 3, "likely": 2, "possible": 1}.get(value, 0)


def add_candidate(bucket: dict, layer: str, label: str, stem: str, strength: str, score: float, detail: str) -> None:
    if not strength:
        return
    existing = bucket[layer].get(label)
    item = {
        "label": label,
        "stem": stem,
        "strength": strength,
        "score": score,
        "detail": detail,
    }
    if not existing:
        bucket[layer][label] = item
        return
    if (strength_rank(strength), score) > (strength_rank(existing["strength"]), safe_float(existing["score"])):
        bucket[layer][label] = item


def downgrade(strength: str, steps: int = 1) -> str:
    order = ["", "possible", "likely", "strong"]
    idx = max(0, order.index(strength) - steps) if strength in order else 0
    return order[idx]


def stem_layer_strength(layer: str, stem: str, strength: str, label: str) -> str:
    if not strength:
        return ""
    if layer == "vocals":
        if stem == "vocals":
            return strength
        if label in {"formant_or_vocoder_vocal", "vocal_synth_hybrid", "pitched_vocal_chop"} and stem in {"other", "guitar", "piano"}:
            return downgrade(strength, 1)
        return downgrade(strength, 2)
    if layer == "drums":
        return strength if stem == "drums" else downgrade(strength, 2)
    if layer == "bass":
        return strength if stem == "bass" else downgrade(strength, 2)
    if layer == "guitar_keys":
        if stem in {"guitar", "piano"}:
            return strength
        if stem == "other":
            return downgrade(strength, 1)
        return downgrade(strength, 2)
    if layer == "sample_fx":
        if stem in {"other", "guitar", "piano", "vocals"}:
            return downgrade(strength, 2)
        return downgrade(strength, 2)
    if layer == "synth":
        if stem in {"other", "guitar", "piano", "bass", "vocals"}:
            return strength
        return downgrade(strength, 2)
    return strength


def synth_status(row: dict, hp_keys: set[tuple[str, str, str, str]]) -> str:
    if key(row) in hp_keys:
        return "strong"
    if row.get("final_label") != "ambiguous" and row.get("source_kind_support") in {"strong_support", "soft_support"}:
        return "likely"
    if row.get("decision") in {"needs_review_or_more_data", "ambiguous_family_only"} and row.get("specialist_strength") in {"medium", "strong"}:
        return "possible"
    return ""


def build_rows(source_rows: list[dict], synth_rows: list[dict], hp_rows: list[dict]) -> list[dict]:
    hp_keys = {key(row) for row in hp_rows}
    synth_by_key = {key(row): row for row in synth_rows}
    grouped: dict[tuple[str, str, str], dict] = {}

    for row in source_rows:
        tkey = time_key(row)
        bucket = grouped.setdefault(
            tkey,
            {
                "track": row["track"],
                "start": norm_time(row["start"]),
                "end": norm_time(row["end"]),
                "layers": {layer: {} for layer in LAYER_ORDER},
                "stems": set(),
            },
        )
        bucket["stems"].add(row["stem"])
        for item in parse_top_source_kinds(row.get("top_source_kinds", "")):
            layer = LABEL_TO_LAYER.get(item["label"])
            if not layer:
                continue
            strength = candidate_strength(item)
            strength = stem_layer_strength(layer, row["stem"], strength, item["label"])
            add_candidate(
                bucket["layers"],
                layer,
                item["label"],
                row["stem"],
                strength,
                item["score"],
                f"{item['score']:.3f}/{item['threshold']:.3f}/{item['status']}",
            )

        flatness = safe_float(row.get("flatness"))
        centroid = safe_float(row.get("centroid"))
        motion = safe_float(row.get("motion_strength"))
        width = safe_float(row.get("width"))
        if row["stem"] in {"other", "vocals", "guitar", "piano"}:
            if flatness >= 0.52 and centroid >= 900:
                add_candidate(
                    bucket["layers"],
                    "noise_fx",
                    "dsp_noisy_texture",
                    row["stem"],
                    "likely",
                    flatness,
                    f"flatness={flatness:.3f} centroid={centroid:.1f}",
                )
            elif flatness >= 0.42 and centroid >= 1200:
                add_candidate(
                    bucket["layers"],
                    "noise_fx",
                    "dsp_noisy_texture",
                    row["stem"],
                    "possible",
                    flatness,
                    f"flatness={flatness:.3f} centroid={centroid:.1f}",
                )
            if motion >= 0.22 and flatness >= 0.32:
                add_candidate(
                    bucket["layers"],
                    "noise_fx",
                    "dsp_glitch_or_transient_motion",
                    row["stem"],
                    "possible",
                    motion,
                    f"motion={motion:.3f} flatness={flatness:.3f}",
                )
            if width >= 1.10 and flatness >= 0.35 and centroid >= 700:
                add_candidate(
                    bucket["layers"],
                    "noise_fx",
                    "dsp_wide_noise_bed",
                    row["stem"],
                    "possible",
                    min(width, 2.0),
                    f"width={width:.3f} flatness={flatness:.3f}",
                )

        srow = synth_by_key.get(key(row))
        if srow:
            status = synth_status(srow, hp_keys)
            label = srow.get("final_label") if srow.get("final_label") != "ambiguous" else srow.get("specialist_label")
            mapped = SYNTH_DETAIL_TO_SOURCE_KIND.get(str(label), str(label))
            layer = "bass" if str(label) in {"synth_bass", "sidechained_synth_bass", "sub_808_synth_bass"} else "synth"
            if str(label) in {"vocal_synth_hybrid", "formant_vocoder"}:
                layer = "vocals"
            add_candidate(
                bucket["layers"],
                layer,
                f"synth_detail:{label}",
                row["stem"],
                status,
                safe_float(srow.get("specialist_conf")),
                f"specialist={srow.get('specialist_conf')} support={srow.get('source_kind_support')}",
            )
            if mapped != label:
                add_candidate(
                    bucket["layers"],
                    layer,
                    mapped,
                    row["stem"],
                    status,
                    safe_float(srow.get("support_score")),
                    f"mapped_from={label}",
                )

    out = []
    for _, bucket in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        row = {
            "track": bucket["track"],
            "start": bucket["start"],
            "end": bucket["end"],
            "stems_seen": "|".join(stem for stem in STEM_ORDER if stem in bucket["stems"]),
        }
        strong_layers = []
        likely_layers = []
        possible_layers = []
        for layer in LAYER_ORDER:
            items = sorted(bucket["layers"][layer].values(), key=lambda x: (strength_rank(x["strength"]), x["score"]), reverse=True)
            items = items[:8]
            strong = [item for item in items if item["strength"] == "strong"]
            likely = [item for item in items if item["strength"] == "likely"]
            possible = [item for item in items if item["strength"] == "possible"]
            if items:
                if strong:
                    strong_layers.append(layer)
                elif likely:
                    likely_layers.append(layer)
                elif possible:
                    possible_layers.append(layer)
            row[layer] = "|".join(f"{item['strength']}:{item['label']}@{item['stem']}:{item['score']:.3f}" for item in items)
            row[f"{layer}_primary_strength"] = items[0]["strength"] if items else ""
        row["strong_layers"] = "|".join(strong_layers)
        row["likely_layers"] = "|".join(likely_layers)
        row["possible_layers"] = "|".join(possible_layers)
        row["strong_layer_count"] = str(len(strong_layers))
        row["broad_layer_count"] = str(len(strong_layers) + len(likely_layers) + len(possible_layers))
        out.append(row)
    return out


def fmt_cell(value: str) -> str:
    if not value:
        return "<span class='empty'>-</span>"
    parts = []
    for raw in value.split("|"):
        try:
            strength, rest = raw.split(":", 1)
            label_stem, score = rest.rsplit(":", 1)
            label, stem = label_stem.rsplit("@", 1)
        except ValueError:
            parts.append(html.escape(raw))
            continue
        cls = html.escape(strength)
        parts.append(f"<div class='{cls}'><b>{html.escape(strength)}</b> {html.escape(label)} <small>{html.escape(stem)} {html.escape(score)}</small></div>")
    return "".join(parts)


def write_outputs(rows: list[dict], out_csv: Path, out_html: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    layer_counts = Counter()
    strength_counts = Counter()
    for row in rows:
        for layer in LAYER_ORDER:
            if row[layer]:
                layer_counts[layer] += 1
                strength_counts[row[f"{layer}_primary_strength"] or "weak"] += 1

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
    sections = []
    for track, track_rows in sorted(by_track.items()):
        trs = []
        for row in sorted(track_rows, key=lambda r: float(r["start"])):
            cells = "".join(f"<td>{fmt_cell(row[layer])}</td>" for layer in LAYER_ORDER)
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><b>strong</b>: {html.escape(row['strong_layers'] or '-')}<br>"
                f"<b>likely</b>: {html.escape(row['likely_layers'] or '-')}<br>"
                f"<b>possible</b>: {html.escape(row['possible_layers'] or '-')}<br>"
                f"<small>{row['strong_layer_count']} strong / {row['broad_layer_count']} broad</small></td>"
                f"{cells}"
                "</tr>"
            )
        header = "".join(f"<th>{html.escape(layer)}</th>" for layer in LAYER_ORDER)
        sections.append(f"<section><h2>{html.escape(track)}</h2><table><tr><th>Time</th><th>Active</th>{header}</tr>{''.join(trs)}</table></section>")

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Broad Multi-Layer Timeline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
td {{ min-width: 150px; }}
.strong {{ background: #dff3e6; padding: 3px; margin: 2px 0; }}
.likely {{ background: #fff2c8; padding: 3px; margin: 2px 0; }}
.possible {{ background: #f3f3f3; padding: 3px; margin: 2px 0; color: #444; }}
.empty {{ color: #888; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1000px; }}
</style>
<h1>Broad Multi-Layer Timeline</h1>
<p class="note">This view intentionally keeps broad candidates. It separates strong / likely / possible instead of forcing one final label per segment. Use this as the main exploratory layer map; use the high-precision shortlist only for stronger claims.</p>
{count_table("Layer Coverage", layer_counts)}
{count_table("Primary Strength Coverage", strength_counts)}
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")
    print(f"wrote: {out_csv}")
    print(f"wrote: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-kind", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind_merged_v3_cached_batch.csv"))
    parser.add_argument("--synth", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v4_cached_aligned_strict_texture.csv"))
    parser.add_argument("--high-precision", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_high_precision_v4_strict.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline.html"))
    args = parser.parse_args()

    rows = build_rows(read_rows(args.source_kind), read_rows(args.synth), read_rows(args.high_precision))
    if not rows:
        raise SystemExit("No rows to render.")
    write_outputs(rows, args.out_csv, args.out_html)


if __name__ == "__main__":
    main()
