from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_NAMES = {
    "source_family": "Likely Source",
    "reverb": "Space / Reverb",
    "distortion": "Distortion / Texture",
    "filter": "Filter",
    "filter_presence": "Filter Presence",
    "filter_motion_type": "Filter Motion",
    "stereo": "Stereo Width",
    "motion": "Motion",
    "motion_presence": "Motion Presence",
}

LABEL_TEXT = {
    "bass": "bass-like low source",
    "guitar_like": "guitar-like plucked or string texture",
    "synth": "synth-like electronic source",
    "vocal_like": "vocal-like tonal source",
    "dry": "dry or very short ambience",
    "short_room": "short room-style reverb",
    "long_hall": "long hall-style reverb",
    "washed_out": "washed-out, long ambient reverb",
    "none": "none detected",
    "mild_saturation": "mild saturation",
    "heavy_distortion": "heavy distortion",
    "crushed": "crushed or bit-reduced texture",
    "filtered": "filtering is likely present",
    "dynamic": "dynamic filter movement is likely",
    "static": "mostly static",
    "mono": "mono or center-focused",
    "medium": "moderate stereo width",
    "wide": "wide stereo image",
    "motion": "audible movement is likely",
}


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_conf(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def describe_top(label: str) -> str:
    return LABEL_TEXT.get(label, label.replace("_", " "))


def confidence_note(confidence: float | None) -> str:
    if confidence is None:
        return "no confidence score available"
    if confidence >= 0.80:
        return "high confidence"
    if confidence >= 0.55:
        return "medium confidence"
    return "low confidence"


def build_markdown(report: dict) -> str:
    predictions = report["predictions"]
    audio = report.get("audio", "unknown")

    lines = [
        "# Sound Design Profiling Report",
        "",
        f"**Audio:** `{audio}`",
        "",
        "## Summary",
    ]

    summary_parts = []
    for key in ["source_family", "reverb", "distortion", "filter_presence", "filter_motion_type", "stereo", "motion_presence"]:
        if key not in predictions:
            continue
        top = predictions[key][0]
        summary_parts.append(f"- **{DISPLAY_NAMES.get(key, key)}:** {describe_top(top['label'])} ({fmt_conf(top['confidence'])})")
    lines.extend(summary_parts or ["- No predictions available."])

    lines.extend(["", "## Detail"])
    for key, values in predictions.items():
        lines.append("")
        lines.append(f"### {DISPLAY_NAMES.get(key, key)}")
        for item in values:
            lines.append(f"- {describe_top(item['label'])}: `{fmt_conf(item['confidence'])}`")

    lines.extend(["", "## Interpretation"])
    interp = []
    if "source_family" in predictions:
        top = predictions["source_family"][0]
        interp.append(f"The source is most consistent with a **{describe_top(top['label'])}** ({confidence_note(top['confidence'])}).")
        if len(predictions["source_family"]) > 1:
            alt = predictions["source_family"][1]
            if alt["confidence"] is not None and alt["confidence"] > 0.25:
                interp.append(f"A secondary possibility is **{describe_top(alt['label'])}**, so this clip may sit between source families.")

    if "reverb" in predictions:
        top = predictions["reverb"][0]
        if top["label"] != "dry":
            interp.append(f"The space profile suggests **{describe_top(top['label'])}**.")
        else:
            interp.append("No strong reverb tail is detected.")

    if "distortion" in predictions:
        top = predictions["distortion"][0]
        if top["label"] != "none":
            interp.append(f"The texture likely includes **{describe_top(top['label'])}**.")
        else:
            interp.append("No clear distortion class is detected.")

    if "filter_presence" in predictions:
        top = predictions["filter_presence"][0]
        if top["label"] == "filtered":
            interp.append("Filtering is likely present, but exact cutoff values are not inferred.")
        else:
            interp.append("The model does not detect strong filtering.")

    if "filter_motion_type" in predictions:
        top = predictions["filter_motion_type"][0]
        if top["label"] == "dynamic":
            interp.append("There is evidence of dynamic filter movement.")

    if "stereo" in predictions:
        top = predictions["stereo"][0]
        interp.append(f"The stereo image appears **{describe_top(top['label'])}**.")

    if "motion_presence" in predictions:
        top = predictions["motion_presence"][0]
        if top["label"] == "motion":
            interp.append("The clip contains audible movement such as pumping, filtering, or other temporal change.")
        else:
            interp.append("The clip appears mostly static over time.")

    lines.extend([f"- {line}" for line in interp])

    lines.extend(
        [
            "",
            "## Caution",
            "- This report estimates broad categorical production attributes.",
            "- It does not recover exact VST plugins, presets, knob values, or effect-chain order.",
            "- Low-confidence alternatives should be treated as hints, not facts.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str) -> str:
    # Tiny project-local renderer: enough for this report without adding deps.
    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Sound Design Profiling Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:920px;margin:32px auto;line-height:1.5;color:#161616}code{background:#f2f2f2;padding:2px 4px}li{margin:4px 0}h1,h2,h3{line-height:1.2}</style>",
    ]
    in_ul = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            continue
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            item = line[2:].replace("**", "").replace("`", "")
            html_lines.append(f"<li>{item}</li>")
        else:
            html_lines.append(f"<p>{line.replace('**', '').replace('`', '')}</p>")
    if in_ul:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    report = load_report(args.input)
    markdown = build_markdown(report)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown, encoding="utf-8")
        print(f"wrote: {args.out_md}")
    else:
        print(markdown)

    if args.out_html:
        args.out_html.parent.mkdir(parents=True, exist_ok=True)
        args.out_html.write_text(markdown_to_html(markdown), encoding="utf-8")
        print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()

