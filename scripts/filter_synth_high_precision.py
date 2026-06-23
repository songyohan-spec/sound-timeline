from __future__ import annotations

import argparse
import csv
import html
from collections import Counter
from pathlib import Path


BASS_LABELS = {"synth_bass", "sidechained_synth_bass", "sub_808_synth_bass"}
TEXTURE_LABELS = {"granular_texture", "wavetable_noise", "bitcrushed_synth_lead", "fuzzy_lofi_synth"}
VOCAL_SYNTH_LABELS = {"vocal_synth_hybrid", "formant_vocoder"}
HARMONIC_STEMS = {"vocals", "other", "guitar", "piano"}
LEAD_PAD_LABELS = {
    "supersaw_stack",
    "synth_pad_wash",
    "digital_synth_lead",
    "synth_pluck_bell",
    "arpeggio_sequence",
    "synth_flute_pipe",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def key(row: dict) -> tuple[str, str, str, str]:
    return row["track"], row["stem"], str(float(row["start"])), str(float(row["end"]))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def keep_reason(row: dict, metrics: dict | None) -> tuple[bool, str]:
    label = row["final_label"]
    stem = row["stem"]
    support = row["source_kind_support"]
    conf = safe_float(row["specialist_conf"])
    support_score = safe_float(row["support_score"])
    centroid = safe_float((metrics or {}).get("centroid"))
    flatness = safe_float((metrics or {}).get("flatness"))
    motion = safe_float((metrics or {}).get("motion_strength"))

    if label == "ambiguous":
        return False, "ambiguous"
    if support not in {"strong_support", "soft_support"}:
        return False, f"weak_support:{support}"
    if support == "soft_support" and conf < 0.70:
        return False, f"soft_low_conf:{conf:.3f}"
    if support == "strong_support" and conf < 0.58:
        return False, f"strong_low_conf:{conf:.3f}"

    if label in BASS_LABELS:
        if stem != "bass":
            return False, f"bass_label_on_{stem}"
        if centroid and centroid > 650:
            return False, f"bass_centroid_too_high:{centroid:.1f}"
        if support_score < 0.45:
            return False, f"bass_support_low:{support_score:.3f}"
        return True, "bass_high_precision"

    if label in TEXTURE_LABELS:
        if support != "strong_support":
            return False, "texture_requires_strong_support"
        if support_score < 0.30:
            return False, f"texture_support_low:{support_score:.3f}"
        if conf < 0.65:
            return False, f"texture_conf_low:{conf:.3f}"
        if flatness and flatness < 0.08 and label in {"granular_texture", "wavetable_noise"}:
            return False, f"texture_too_tonal:{flatness:.3f}"
        return True, "texture_high_precision"

    if label in VOCAL_SYNTH_LABELS:
        if stem not in HARMONIC_STEMS:
            return False, f"vocal_synth_on_{stem}"
        if conf < 0.55 and support_score < 0.45:
            return False, "vocal_synth_weak_joint_score"
        return True, "vocal_synth_high_precision"

    if label in LEAD_PAD_LABELS:
        if stem == "drums":
            return False, "lead_pad_on_drums"
        if conf < 0.62:
            return False, f"lead_pad_conf_low:{conf:.3f}"
        if support_score < 0.32:
            return False, f"lead_pad_support_low:{support_score:.3f}"
        if label == "arpeggio_sequence" and motion and motion < 0.08:
            return False, f"arpeggio_motion_low:{motion:.3f}"
        return True, "lead_pad_high_precision"

    if conf >= 0.68 and support_score >= 0.40:
        return True, "generic_high_precision"
    return False, "generic_low_joint_score"


def merge_regions(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: (r["track"], r["stem"], r["final_label"], safe_float(r["start"])))
    regions = []
    for row in rows:
        start = safe_float(row["start"])
        end = safe_float(row["end"])
        if (
            regions
            and regions[-1]["track"] == row["track"]
            and regions[-1]["stem"] == row["stem"]
            and regions[-1]["final_label"] == row["final_label"]
            and abs(safe_float(regions[-1]["end"]) - start) <= 0.05
        ):
            regions[-1]["end"] = row["end"]
            regions[-1]["segments"] += 1
            regions[-1]["max_conf"] = max(regions[-1]["max_conf"], safe_float(row["specialist_conf"]))
            regions[-1]["max_support"] = max(regions[-1]["max_support"], safe_float(row["support_score"]))
        else:
            regions.append(
                {
                    "track": row["track"],
                    "stem": row["stem"],
                    "start": row["start"],
                    "end": row["end"],
                    "final_label": row["final_label"],
                    "segments": 1,
                    "max_conf": safe_float(row["specialist_conf"]),
                    "max_support": safe_float(row["support_score"]),
                    "reason": row["hp_reason"],
                }
            )
    return regions


def write_html(rows: list[dict], regions: list[dict], out_html: Path) -> None:
    label_counts = Counter(row["final_label"] for row in rows)
    stem_counts = Counter(row["stem"] for row in rows)
    region_label_counts = Counter(row["final_label"] for row in regions)

    def table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table>"

    region_rows = []
    for row in regions:
        region_rows.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td>{html.escape(row['final_label'])}</td>"
            f"<td>{row['segments']}</td>"
            f"<td>{row['max_conf']:.3f}</td>"
            f"<td>{row['max_support']:.3f}</td>"
            f"<td>{html.escape(row['reason'])}</td>"
            "</tr>"
        )

    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td>{html.escape(row['final_label'])}</td>"
            f"<td>{row['specialist_conf']}</td>"
            f"<td>{html.escape(row['source_kind_support'])}</td>"
            f"<td>{row['support_score']}</td>"
            f"<td>{html.escape(row['hp_reason'])}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>High Precision Synth Shortlist</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>High Precision Synth Shortlist</h1>
<p class="note">This is a stricter layer on top of the usable synth candidates. It intentionally sacrifices recall to reduce noisy candidates.</p>
<p>Segments: {len(rows)} / Regions: {len(regions)}</p>
{table("Segment Labels", label_counts)}
{table("Region Labels", region_label_counts)}
{table("Stems", stem_counts)}
<h2>Regions</h2>
<table><tr><th>Track</th><th>Stem</th><th>Time</th><th>Label</th><th>Segments</th><th>Max Conf</th><th>Max Support</th><th>Reason</th></tr>{''.join(region_rows)}</table>
<h2>Segments</h2>
<table><tr><th>Track</th><th>Stem</th><th>Time</th><th>Label</th><th>Conf</th><th>Support</th><th>Support Score</th><th>Reason</th></tr>{''.join(detail_rows)}</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v4_cached_aligned_strict_texture.csv"))
    parser.add_argument("--source-kind", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind_merged_v3_cached_batch.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_high_precision_v4_strict.csv"))
    parser.add_argument("--out-regions", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_high_precision_v4_strict_regions.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_high_precision_v4_strict.html"))
    args = parser.parse_args()

    metrics = {key(row): row for row in read_rows(args.source_kind)}
    kept = []
    rejected = Counter()
    for row in read_rows(args.ensemble):
        ok, reason = keep_reason(row, metrics.get(key(row)))
        if ok:
            out = dict(row)
            out["hp_reason"] = reason
            kept.append(out)
        else:
            rejected[reason.split(":", 1)[0]] += 1

    regions = merge_regions(kept)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(kept[0].keys()) if kept else ["track"])
        writer.writeheader()
        writer.writerows(kept)
    with args.out_regions.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(regions[0].keys()) if regions else ["track"])
        writer.writeheader()
        writer.writerows(regions)
    write_html(kept, regions, args.out_html)
    print(f"kept segments: {len(kept)}")
    print(f"kept regions: {len(regions)}")
    print("rejected:", dict(rejected.most_common(12)))
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_regions}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
