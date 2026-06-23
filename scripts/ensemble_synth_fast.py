from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


ACTIVE = {"medium", "strong"}

SOURCE_KIND_SUPPORT = {
    "synth_pad_wash": {"synth_pad_or_wash", "supersaw_or_bright_synth_stack"},
    "supersaw_stack": {"supersaw_or_bright_synth_stack", "synth_pad_or_wash"},
    "digital_synth_lead": {"digital_synth_lead", "arpeggio_or_sequence_synth"},
    "bitcrushed_synth_lead": {"bitcrushed_or_aliasing_synth", "digital_synth_lead", "fuzzy_distorted_synth"},
    "synth_pluck_bell": {"synth_pluck_or_bell", "digital_synth_lead", "arpeggio_or_sequence_synth"},
    "arpeggio_sequence": {"arpeggio_or_sequence_synth", "synth_pluck_or_bell"},
    "granular_texture": {"granular_or_resampled_synth", "wavetable_noise_synth"},
    "wavetable_noise": {"wavetable_noise_synth", "granular_or_resampled_synth"},
    "fuzzy_lofi_synth": {"fuzzy_distorted_synth", "bitcrushed_or_aliasing_synth"},
    "synth_flute_pipe": {"synth_pluck_or_bell", "digital_synth_lead"},
    "vocal_synth_hybrid": {"vocal_synth_hybrid", "formant_or_vocoder_vocal"},
    "formant_vocoder": {"formant_or_vocoder_vocal", "vocal_synth_hybrid"},
    "synth_bass": {"synth_bass", "sub_or_808_bass"},
    "sidechained_synth_bass": {"sidechained_bass_pulse", "synth_bass"},
    "sub_808_synth_bass": {"sub_or_808_bass", "synth_bass", "distorted_bass"},
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def key(row: dict) -> tuple[str, str, str, str]:
    return row["track"], row["stem"], str(float(row["start"])), str(float(row["end"]))


def parse_top_source_kinds(value: str) -> dict[str, tuple[float, str]]:
    out = {}
    for raw in str(value or "").split(";"):
        raw = raw.strip()
        if not raw or ":" not in raw:
            continue
        label, rest = raw.split(":", 1)
        parts = rest.split("/")
        if len(parts) < 3:
            continue
        try:
            score = float(parts[0])
        except ValueError:
            continue
        out[label.strip()] = (score, parts[2].strip())
    return out


def support_for(row: dict, source_row: dict | None) -> tuple[str, str, float]:
    label = row["synth_label_top"]
    expected = SOURCE_KIND_SUPPORT.get(label, set())
    if not source_row:
        return "missing_source_kind", "", 0.0
    source_labels = parse_top_source_kinds(source_row.get("top_source_kinds", ""))
    detected = set(str(source_row.get("detected_source_kinds", "")).split("|"))
    matched = []
    best = 0.0
    for candidate in expected:
        if candidate in source_labels:
            score, status = source_labels[candidate]
            best = max(best, score)
            matched.append(f"{candidate}:{score:.3f}/{status}")
        elif candidate in detected:
            matched.append(f"{candidate}:detected")
    if any("/detected" in item or item.endswith(":detected") for item in matched):
        return "strong_support", "|".join(matched), best
    if matched:
        return "soft_support", "|".join(matched), best
    dsp_support, dsp_reason, dsp_score = dsp_support_for(row, source_row)
    if dsp_support:
        return dsp_support, dsp_reason, dsp_score
    if str(source_row.get("detected_source_groups", "")).find("synth") >= 0 and row["synth_family_top"] == "synth":
        return "family_support", str(source_row.get("detected_source_groups", "")), best
    return "unsupported", "", best


def safe_float(row: dict | None, key: str, default: float = 0.0) -> float:
    if not row:
        return default
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def dsp_support_for(row: dict, source_row: dict | None) -> tuple[str, str, float]:
    if not source_row:
        return "", "", 0.0
    label = row["synth_label_top"]
    stem = row["stem"]
    centroid = safe_float(source_row, "centroid")
    flatness = safe_float(source_row, "flatness")
    motion = safe_float(source_row, "motion_strength")
    width = min(safe_float(source_row, "width"), 2.0)
    tonal = flatness <= 0.28
    noisy = flatness >= 0.34
    wide = width >= 0.48
    bright = centroid >= 900
    very_bright = centroid >= 1500
    lowish = centroid <= 420
    moving = motion >= 0.16
    stable = motion <= 0.18

    if stem == "drums":
        return "", "", 0.0
    if label == "synth_pad_wash" and wide and tonal and stable and centroid >= 350:
        return "dsp_support", f"wide={width:.3f}|tonal_flatness={flatness:.3f}|stable_motion={motion:.3f}", 0.42
    if label == "supersaw_stack" and wide and tonal and bright:
        return "dsp_support", f"wide={width:.3f}|bright_centroid={centroid:.1f}|tonal_flatness={flatness:.3f}", 0.44
    if label == "digital_synth_lead" and tonal and very_bright:
        return "dsp_support", f"bright_centroid={centroid:.1f}|tonal_flatness={flatness:.3f}", 0.40
    if label == "bitcrushed_synth_lead" and very_bright and noisy:
        return "dsp_support", f"bright_centroid={centroid:.1f}|noisy_flatness={flatness:.3f}", 0.40
    if label == "arpeggio_sequence" and bright and moving:
        return "dsp_support", f"bright_centroid={centroid:.1f}|motion={motion:.3f}", 0.40
    if label == "vocal_synth_hybrid" and stem in {"vocals", "other"} and tonal and centroid >= 450:
        return "dsp_support", f"voice_stem={stem}|tonal_flatness={flatness:.3f}|centroid={centroid:.1f}", 0.40
    if label == "formant_vocoder" and stem in {"vocals", "other", "guitar", "piano"} and tonal and bright:
        return "dsp_support", f"formant_like_stem={stem}|centroid={centroid:.1f}|flatness={flatness:.3f}", 0.40
    if label in {"synth_bass", "sidechained_synth_bass", "sub_808_synth_bass"} and stem == "bass" and lowish:
        return "dsp_support", f"bass_stem_low_centroid={centroid:.1f}|motion={motion:.3f}", 0.42
    return "", "", 0.0


def decision(row: dict, support: str, min_soft_conf: float, min_family_conf: float) -> str:
    if row["strength"] not in ACTIVE:
        return "ignore"
    specialist_conf = float(row.get("synth_label_conf", row.get("specialist_conf", 0.0)) or 0.0)
    if support == "strong_support":
        return "use_pseudo_label"
    if support == "soft_support" and specialist_conf >= min_soft_conf:
        return "use_pseudo_label"
    if support == "dsp_support" and specialist_conf >= min_soft_conf:
        return "use_weak_pseudo_label"
    if row["strength"] == "strong" and support == "family_support":
        if specialist_conf < min_family_conf:
            return "ambiguous_family_only"
        return "use_weak_pseudo_label"
    if support == "family_support":
        return "ambiguous_family_only"
    return "needs_review_or_more_data"


def write_html(rows: list[dict], out_html: Path) -> None:
    decision_counts = Counter(row["decision"] for row in rows)
    label_counts = Counter(row["final_label"] for row in rows if row["final_label"] != "ambiguous")
    counts = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in decision_counts.most_common())
    labels = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in label_counts.most_common(18))
    details = []
    for row in rows:
        details.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td>{html.escape(row['specialist_label'])} ({row['specialist_conf']})</td>"
            f"<td>{html.escape(row['source_kind_support'])}</td>"
            f"<td>{html.escape(row['support_matches'])}</td>"
            f"<td class='{html.escape(row['decision'])}'>{html.escape(row['decision'])}</td>"
            f"<td>{html.escape(row['final_label'])}</td>"
            "</tr>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Fast Synth Ensemble</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.use_pseudo_label, .use_weak_pseudo_label {{ background: #e8f7ed; }}
.ambiguous_family_only {{ background: #fff7df; }}
.needs_review_or_more_data {{ background: #ffe9dc; }}
.ignore {{ background: #f4f4f4; color: #777; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Fast Synth Ensemble</h1>
<p class="note">Combines Synth Specialist with the existing stem source-kind model. This is the practical fast agreement layer while CLAP teacher remains too slow/unstable for full local batches.</p>
<h2>Decisions</h2>
<table><tr><th>Decision</th><th>Count</th></tr>{counts}</table>
<h2>Usable Label Counts</h2>
<table><tr><th>Label</th><th>Count</th></tr>{labels}</table>
<h2>Details</h2>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Specialist</th><th>Source-Kind Support</th><th>Matches</th><th>Decision</th><th>Final Label</th></tr>
{''.join(details)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist.csv"))
    parser.add_argument("--source-kind", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble.html"))
    parser.add_argument("--min-soft-conf", type=float, default=0.50)
    parser.add_argument("--min-family-conf", type=float, default=0.64)
    args = parser.parse_args()

    source_rows = {key(row): row for row in read_rows(args.source_kind)}
    out = []
    for row in read_rows(args.synth):
        source_row = source_rows.get(key(row))
        support, matches, support_score = support_for(row, source_row)
        final_decision = decision(row, support, args.min_soft_conf, args.min_family_conf)
        final_label = row["synth_label_top"] if final_decision in {"use_pseudo_label", "use_weak_pseudo_label"} else "ambiguous"
        out.append(
            {
                "track": row["track"],
                "stem": row["stem"],
                "start": row["start"],
                "end": row["end"],
                "specialist_family": row["synth_family_top"],
                "specialist_label": row["synth_label_top"],
                "specialist_conf": row["synth_label_conf"],
                "specialist_strength": row["strength"],
                "source_kind_support": support,
                "support_matches": matches,
                "support_score": round(support_score, 6),
                "decision": final_decision,
                "final_label": final_label,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    write_html(out, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
