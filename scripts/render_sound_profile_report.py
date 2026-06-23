from __future__ import annotations

import argparse
import json
from pathlib import Path


FAMILY_NAMES = {
    "vocal_derived": "Vocal-Derived",
    "synth_derived": "Synth-Derived",
    "guitar_derived": "Guitar-Derived",
    "fx_texture": "FX / Texture",
    "processing_space": "Processing / Space",
}


LABEL_DESCRIPTIONS = {
    "filtered_guitar_loop": "a guitar-like loop with filtered or muted tone",
    "chorus_guitar_wash": "a wide, dreamy guitar wash",
    "pitched_vocal_chop": "a melodic chopped vocal-like sample",
    "hard_tuned_vocal_lead": "a tuned or robotic vocal-like lead",
    "formant_shifted_vocal_chop": "a vocal chop with formant-like character shift",
    "granular_vocal_smear": "a blurred or granular vocal texture",
    "breathy_vocal_pad": "an airy vocal pad texture",
    "airy_synth_pad": "a soft atmospheric synth pad",
    "supersaw_wide_pad": "a wide detuned saw-style synth pad",
    "filtered_synth_pluck": "a filtered short synth pluck",
    "glassy_fm_bell": "a bright metallic FM-like bell",
    "bitcrushed_synth_lead": "a crushed digital synth lead",
    "noisy_wavetable_texture": "a grainy digital synth texture",
    "vinyl_noise_texture": "a crackly noise/record texture",
    "glitch_grain_texture": "a glitchy granular texture",
    "tape_noise_bed": "a tape-hiss style background bed",
    "sidechain_pumping": "a pumping or ducking movement",
    "chopped_retriggered_envelope": "a gated or retriggered rhythmic envelope",
    "long_reverb_wash": "a long washed reverb space",
    "wide_chorus_widening": "a wide stereo chorus/widening treatment",
    "delay_throw_tail": "a delayed echo tail",
    "dry_close": "a close and relatively dry presentation",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def top_items(palette: dict, family: str, n: int = 3) -> list[dict]:
    return palette.get("families", {}).get(family, [])[:n]


def first(palette: dict, family: str) -> dict | None:
    items = top_items(palette, family, 1)
    return items[0] if items else None


def score(item: dict | None) -> float:
    return float(item.get("score", 0.0)) if item else 0.0


def label(item: dict | None) -> str:
    return str(item.get("label", "unknown")) if item else "unknown"


def confidence_tier(item: dict | None, reference: float | None = None) -> str:
    value = score(item)
    if reference is not None and reference > 0:
        ratio = value / reference
        if ratio >= 0.65:
            return "primary"
        if ratio >= 0.15:
            return "secondary"
        return "weak"
    if value >= 0.05:
        return "primary"
    if value >= 0.005:
        return "secondary"
    return "weak"


def describe_label(name: str) -> str:
    return LABEL_DESCRIPTIONS.get(name, name.replace("_", " "))


def rf_top(segment_report: dict, key: str) -> tuple[str, float] | None:
    segments = segment_report.get("segments", [])
    if not segments:
        return None
    values = segments[0].get("predictions", {}).get(key, [])
    if not values:
        return None
    return str(values[0]["label"]), float(values[0]["confidence"])


def render_markdown(palette: dict, title: str, segment_report: dict | None = None) -> str:
    top_overall = palette.get("top_overall", [])
    primary = top_overall[0] if top_overall else None
    primary_score = score(primary)

    guitar = first(palette, "guitar_derived")
    vocal = first(palette, "vocal_derived")
    synth = first(palette, "synth_derived")
    fx = first(palette, "fx_texture")
    processing = first(palette, "processing_space")

    lines = [
        f"# Sound Profile Report: {title}",
        "",
        "## Main Read",
        "",
    ]

    if primary:
        lines.append(
            f"- **Primary candidate:** `{label(primary)}` ({describe_label(label(primary))}). "
            f"CLAP palette score `{score(primary):.6f}`."
        )
    else:
        lines.append("- No primary candidate found.")

    if guitar and confidence_tier(guitar, primary_score) in {"primary", "secondary"}:
        lines.append(f"- **Guitar-derived cue:** `{label(guitar)}` - {describe_label(label(guitar))}.")
    if vocal and confidence_tier(vocal, primary_score) in {"primary", "secondary"}:
        lines.append(f"- **Vocal-derived cue:** `{label(vocal)}` - {describe_label(label(vocal))}.")
    if synth and confidence_tier(synth, primary_score) in {"primary", "secondary"}:
        lines.append(f"- **Synth-derived cue:** `{label(synth)}` - {describe_label(label(synth))}.")
    if processing and confidence_tier(processing, primary_score) in {"primary", "secondary"}:
        lines.append(f"- **Processing / movement cue:** `{label(processing)}` - {describe_label(label(processing))}.")
    if fx and confidence_tier(fx, primary_score) in {"primary", "secondary"}:
        lines.append(f"- **FX / texture cue:** `{label(fx)}` - {describe_label(label(fx))}.")

    lines.extend(["", "## Layer Hypothesis", ""])

    hypotheses = []
    if primary and label(primary) == "filtered_guitar_loop":
        hypotheses.append(
            "The most plausible foreground layer is a **filtered guitar-loop-like motif**, rather than a clean synth pad."
        )
    if processing and label(processing) == "sidechain_pumping":
        hypotheses.append(
            "There is a noticeable **pumping / ducking movement** cue. This may be sidechain-like movement or a rhythmic envelope effect."
        )
    if vocal and label(vocal) in {"pitched_vocal_chop", "hard_tuned_vocal_lead", "formant_shifted_vocal_chop"}:
        hypotheses.append(
            f"A secondary vocal-derived reading is present: **{label(vocal)}**. Treat this as a possible layer or timbral resemblance, not confirmed separation."
        )
    if synth and confidence_tier(synth, primary_score) == "weak":
        hypotheses.append(
            "Synth-pad candidates are weak compared with the guitar/vocal/motion cues."
        )

    lines.extend([f"- {item}" for item in hypotheses] or ["- No stable layer hypothesis could be formed."])

    lines.extend(["", "## Palette Ranking", ""])
    lines.append("| Rank | Family | Label | Score | Prompt |")
    lines.append("|---:|---|---|---:|---|")
    for idx, item in enumerate(top_overall[:10], start=1):
        lines.append(
            f"| {idx} | {FAMILY_NAMES.get(item.get('family', ''), item.get('family', ''))} | `{item['label']}` | {float(item['score']):.6f} | {item.get('prompt', '')} |"
        )

    if segment_report:
        lines.extend(["", "## RF/DSP Cross-Check", ""])
        checks = [
            ("source_family", "Source"),
            ("reverb", "Reverb"),
            ("distortion", "Distortion"),
            ("stereo", "Stereo"),
            ("motion_presence", "Motion"),
        ]
        for key, display in checks:
            value = rf_top(segment_report, key)
            if value:
                lines.append(f"- **{display}:** `{value[0]}` ({value[1]:.3f})")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- Use this report as a **ranked sound-palette hypothesis**, not a definitive transcription.",
            "- The strongest current read is the first-ranked palette item plus any secondary cue whose score is not tiny relative to the primary.",
            "- Do not infer exact plugin, preset, oscillator, knob value, or effect-chain order.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str) -> str:
    html = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Sound Profile Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1050px;margin:32px auto;line-height:1.5;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}code{background:#f3f3f3;padding:2px 4px}li{margin:5px 0}</style>",
    ]
    in_ul = False
    in_table = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            if in_ul:
                html.append("</ul>")
                in_ul = False
            if in_table:
                html.append("</tbody></table>")
                in_table = False
            continue
        if line.startswith("# "):
            html.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("|"):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if not in_table:
                html.append("<table><tbody>")
                in_table = True
            tag = "th" if cells[0] in {"Rank", "Attribute"} else "td"
            html.append("<tr>" + "".join(f"<{tag}>{cell.replace('`', '')}</{tag}>" for cell in cells) + "</tr>")
        elif line.startswith("- "):
            if not in_ul:
                html.append("<ul>")
                in_ul = True
            item = line[2:].replace("**", "").replace("`", "")
            html.append(f"<li>{item}</li>")
        else:
            html.append(f"<p>{line.replace('**', '').replace('`', '')}</p>")
    if in_ul:
        html.append("</ul>")
    if in_table:
        html.append("</tbody></table>")
    return "\n".join(html)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--title", default="External Clip")
    parser.add_argument("--out-md", type=Path, default=Path("outputs/sound_profile_report.md"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/sound_profile_report.html"))
    args = parser.parse_args()

    palette = load_json(args.palette)
    segment_report = load_json(args.segments) if args.segments else None
    markdown = render_markdown(palette, args.title, segment_report)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_html.write_text(markdown_to_html(markdown), encoding="utf-8")
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

