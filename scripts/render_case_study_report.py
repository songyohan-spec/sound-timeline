from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean


FIELDS = ["source", "reverb", "distortion", "stereo", "motion"]


def read_summary_rows(batch_dir: Path) -> list[dict]:
    summary = batch_dir / "summary.csv"
    if summary.exists():
        with summary.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    rows: list[dict] = []
    for path in sorted(batch_dir.glob("*_pseudo_labels.csv")):
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                rows.append(
                    {
                        "file": path.name.replace("_pseudo_labels.csv", ""),
                        "start": row.get("start", ""),
                        "end": row.get("end", ""),
                        "source": row.get("source_primary", ""),
                        "source_conf": row.get("source_primary_conf", ""),
                        "source_alt": row.get("source_alt", ""),
                        "reverb": row.get("reverb_label", ""),
                        "reverb_conf": row.get("reverb_conf", ""),
                        "distortion": row.get("distortion_label", ""),
                        "distortion_conf": row.get("distortion_conf", ""),
                        "stereo": row.get("stereo_label", ""),
                        "stereo_conf": row.get("stereo_conf", ""),
                        "motion": row.get("motion_label", ""),
                        "motion_conf": row.get("motion_conf", ""),
                        "needs_review": row.get("needs_review", ""),
                        "low_confidence_fields": row.get("low_confidence_fields", ""),
                    }
                )
    return rows


def conf(row: dict, field: str) -> float:
    try:
        return float(row.get(f"{field}_conf", ""))
    except ValueError:
        return 0.0


def counts(rows: list[dict], field: str) -> Counter[str]:
    return Counter(row.get(field, "") for row in rows if row.get(field, ""))


def mean_conf(rows: list[dict], field: str) -> float:
    values = [conf(row, field) for row in rows if conf(row, field) > 0]
    return mean(values) if values else 0.0


def consensus_sentence(rows: list[dict], field: str, display: str) -> str:
    counter = counts(rows, field)
    if not counter:
        return f"No {display} estimate was available."
    label, count = counter.most_common(1)[0]
    ratio = count / len(rows)
    avg = mean_conf(rows, field)
    if ratio >= 0.8 and avg >= 0.75:
        strength = "strong"
    elif ratio >= 0.6:
        strength = "moderate"
    else:
        strength = "weak"
    return f"{display}: **{label}** appeared in {count}/{len(rows)} segments; consensus is {strength}, mean confidence {avg:.3f}."


def render_markdown(rows: list[dict], title: str) -> str:
    low_conf = Counter()
    for row in rows:
        for field in row.get("low_confidence_fields", "").split(";"):
            if field:
                low_conf[field] += 1

    lines = [
        f"# Case Study Report: {title}",
        "",
        f"Segments analyzed: **{len(rows)}**",
        "",
        "## Song-Level Pattern",
        "",
        consensus_sentence(rows, "source", "Source"),
        consensus_sentence(rows, "reverb", "Reverb"),
        consensus_sentence(rows, "distortion", "Distortion"),
        consensus_sentence(rows, "stereo", "Stereo"),
        consensus_sentence(rows, "motion", "Motion"),
        "",
        "## Interpretation",
        "",
    ]

    source_top = counts(rows, "source").most_common(1)[0][0] if counts(rows, "source") else "unknown"
    reverb_top = counts(rows, "reverb").most_common(1)[0][0] if counts(rows, "reverb") else "unknown"
    stereo_top = counts(rows, "stereo").most_common(1)[0][0] if counts(rows, "stereo") else "unknown"
    motion_top = counts(rows, "motion").most_common(1)[0][0] if counts(rows, "motion") else "unknown"

    lines.extend(
        [
            f"- The current model repeatedly maps this material toward **{source_top}**, suggesting that the clip family sits closer to processed vocal / synthetic texture space than to clean NSynth-style instruments.",
            f"- The repeated **{reverb_top}** and **{stereo_top}** estimates suggest a spacious, wide production profile, but reverb confidence is low and should be treated as a weak cue.",
            f"- The repeated **{motion_top}** estimate suggests temporal movement, but DSP disagreement means this may include general envelope movement rather than a specific sidechain or filter motion.",
            "",
            "## Reliability",
            "",
        ]
    )

    if low_conf:
        for field, count in low_conf.most_common():
            lines.append(f"- `{field}` was low-confidence in {count}/{len(rows)} segments.")
    else:
        lines.append("- No low-confidence fields were reported.")

    lines.extend(
        [
            "",
            "## Next Modeling Implication",
            "",
            "- Treat this as a **single-song case study**, not a broad validation set.",
            "- The current RF profiler is useful for broad cues like stereo/motion, but source/reverb/distortion need stronger pseudo-label support.",
            "- The next technical step should be adding a text-audio model panel such as CLAP, or building more synthetic examples around processed vocal, vocal chop, washed guitar, and ambient pad textures.",
            "",
            "## Caution",
            "",
            "- This report does not identify exact plugins, presets, stems, or effect-chain order.",
            "- These are weak estimates from a model trained on synthetic/NSynth-derived examples.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_to_html(markdown: str) -> str:
    lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Case Study Report</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:920px;margin:32px auto;line-height:1.5;color:#161616}code{background:#f2f2f2;padding:2px 4px}li{margin:5px 0}h1,h2{line-height:1.2}</style>",
    ]
    in_ul = False
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            if in_ul:
                lines.append("</ul>")
                in_ul = False
            continue
        if line.startswith("# "):
            lines.append(f"<h1>{line[2:].replace('**', '')}</h1>")
        elif line.startswith("## "):
            if in_ul:
                lines.append("</ul>")
                in_ul = False
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("- "):
            if not in_ul:
                lines.append("<ul>")
                in_ul = True
            item = line[2:].replace("**", "").replace("`", "")
            lines.append(f"<li>{item}</li>")
        else:
            lines.append(f"<p>{line.replace('**', '').replace('`', '')}</p>")
    if in_ul:
        lines.append("</ul>")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-dir", type=Path, default=Path("outputs/external_batch_alt2"))
    parser.add_argument("--title", default="External Clip Set")
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    rows = read_summary_rows(args.batch_dir)
    if not rows:
        raise SystemExit(f"No pseudo-label rows found under {args.batch_dir}")

    out_md = args.out_md or (args.batch_dir / "case_study_report.md")
    out_html = args.out_html or (args.batch_dir / "case_study_report.html")
    markdown = render_markdown(rows, args.title)
    out_md.write_text(markdown, encoding="utf-8")
    out_html.write_text(markdown_to_html(markdown), encoding="utf-8")
    print(f"wrote: {out_md}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()

