from __future__ import annotations

import argparse
import json
from pathlib import Path


def top_from_segment(segment_report: dict, key: str) -> tuple[str, float]:
    segments = segment_report.get("segments", [])
    if not segments:
        return "unknown", 0.0
    values = segments[0].get("predictions", {}).get(key, [])
    if not values:
        return "unknown", 0.0
    top = values[0]
    return str(top["label"]), float(top["confidence"])


def top_from_clap(clap_report: dict, group: str) -> tuple[str, float]:
    values = clap_report.get("scores", {}).get(group, [])
    if not values:
        return "unknown", 0.0
    top = values[0]
    return str(top["label"]), float(top["score"])


def source_family(label: str) -> str:
    if label in {"processed_vocal", "vocal_chop", "vocal_like"}:
        return "vocal_processed"
    if label in {"synth", "ambient_pad", "texture_noise"}:
        return "synthetic_texture"
    if label in {"guitar_like", "washed_guitar"}:
        return "guitar_texture"
    return label


def agreement(a: str, b: str) -> str:
    if a == "unknown" or b == "unknown":
        return "unknown"
    return "agree" if source_family(a) == source_family(b) else "disagree"


def confidence_word(conf: float, high: float = 0.75, medium: float = 0.55) -> str:
    if conf >= high:
        return "high"
    if conf >= medium:
        return "medium"
    return "low"


def render_markdown(segment_report: dict, clap_report: dict, title: str) -> str:
    rf_source, rf_source_conf = top_from_segment(segment_report, "source_family")
    rf_reverb, rf_reverb_conf = top_from_segment(segment_report, "reverb")
    rf_dist, rf_dist_conf = top_from_segment(segment_report, "distortion")
    rf_stereo, rf_stereo_conf = top_from_segment(segment_report, "stereo")
    rf_motion, rf_motion_conf = top_from_segment(segment_report, "motion_presence")

    clap_source, clap_source_score = top_from_clap(clap_report, "source")
    clap_space, clap_space_score = top_from_clap(clap_report, "space")
    clap_texture, clap_texture_score = top_from_clap(clap_report, "texture")
    clap_motion, clap_motion_score = top_from_clap(clap_report, "motion")

    source_agreement = agreement(rf_source, clap_source)

    lines = [
        f"# Ensemble Sound Profile: {title}",
        "",
        "## Panel Results",
        "",
        "| Attribute | RF/DSP panel | CLAP semantic panel | Agreement |",
        "|---|---:|---:|---|",
        f"| Source | {rf_source} ({rf_source_conf:.3f}) | {clap_source} ({clap_source_score:.6f}) | {source_agreement} |",
        f"| Space/Reverb | {rf_reverb} ({rf_reverb_conf:.3f}) | {clap_space} ({clap_space_score:.6f}) | hint-only |",
        f"| Texture/Distortion | {rf_dist} ({rf_dist_conf:.3f}) | {clap_texture} ({clap_texture_score:.6f}) | hint-only |",
        f"| Stereo | {rf_stereo} ({rf_stereo_conf:.3f}) | n/a | RF/DSP only |",
        f"| Motion | {rf_motion} ({rf_motion_conf:.3f}) | {clap_motion} ({clap_motion_score:.6f}) | hint-only |",
        "",
        "## Interpretation",
        "",
    ]

    if source_agreement == "agree":
        lines.append(
            f"- Both panels point toward the broader **{source_family(rf_source)}** family. "
            f"RF/DSP says `{rf_source}`, while CLAP says `{clap_source}`."
        )
    else:
        lines.append(
            f"- The source estimate is not stable across panels. RF/DSP says `{rf_source}`, while CLAP says `{clap_source}`."
        )

    if clap_source == "vocal_chop":
        lines.append("- CLAP strongly favors a **vocal chop / processed vocal** interpretation, which is useful for alternative-pop style material.")
    elif clap_source == "processed_vocal":
        lines.append("- CLAP favors a **processed vocal texture** interpretation.")

    if clap_space == "washed_out" or rf_reverb in {"long_hall", "washed_out"}:
        lines.append("- Space cues lean toward a **washed-out / spacious** profile, but RF reverb confidence should still be treated carefully.")
    else:
        lines.append("- Space cues do not strongly indicate a large reverb tail.")

    if rf_stereo == "wide" and rf_stereo_conf >= 0.75:
        lines.append("- The wide stereo estimate is one of the more reliable cues in the current system.")

    if rf_motion == "motion" or clap_motion == "moving":
        lines.append("- Both the task framing and CLAP prompts suggest audible movement, but this does not identify the exact motion type.")

    if clap_texture == "clean" and rf_dist in {"crushed", "heavy_distortion"} and rf_dist_conf < 0.65:
        lines.append("- Distortion is likely not a strong conclusion here: CLAP leans clean while RF distortion confidence is not high.")

    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- **Use as pseudo-label:** `{clap_source}` / `{clap_space}` / `{rf_stereo}` / `{rf_motion}`.",
            "- **Do not use as hard label:** exact distortion class, exact reverb type, exact source identity.",
            "",
            "## Caution",
            "",
            "- This is an ensemble of weak panels, not ground truth.",
            "- CLAP scores are semantic prompt rankings, not calibrated probabilities.",
            "- RF/DSP was trained on NSynth-derived synthetic examples, so real-song confidence can remain low.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str) -> str:
    html = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Ensemble Sound Profile</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:980px;margin:32px auto;line-height:1.5;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f2f2f2}code{background:#f3f3f3;padding:2px 4px}li{margin:5px 0}</style>",
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
            tag = "th" if cells[0] == "Attribute" else "td"
            html.append("<tr>" + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells) + "</tr>")
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
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--clap", type=Path, required=True)
    parser.add_argument("--title", default="External Clip")
    parser.add_argument("--out-md", type=Path, default=Path("outputs/ensemble_report.md"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/ensemble_report.html"))
    args = parser.parse_args()

    segment_report = json.loads(args.segments.read_text(encoding="utf-8"))
    clap_report = json.loads(args.clap.read_text(encoding="utf-8"))
    markdown = render_markdown(segment_report, clap_report, args.title)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(markdown, encoding="utf-8")
    args.out_html.write_text(markdown_to_html(markdown), encoding="utf-8")
    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

