from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter
from pathlib import Path


RISKY_TEXTURE_MARKERS = (
    "granular_texture",
    "granular_or_resampled_synth",
    "wavetable_noise",
    "fuzzy_distorted_synth",
    "fuzzy_lofi_synth",
)

ANCHOR_MARKERS = (
    "supersaw",
    "synth_pluck_bell",
    "synth_pad_wash",
    "synth_pad_or_wash",
    "digital_synth_lead",
    "arpeggio_sequence",
)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_cell(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def label_has_any(label: str, markers: tuple[str, ...]) -> bool:
    return any(marker in label for marker in markers)


def decision_for(row: dict, max_share: float) -> tuple[str, str]:
    labels = split_cell(row.get("labels", ""))
    share = safe_float(row.get("candidate_rms_share"))
    has_anchor = any(label_has_any(label, ANCHOR_MARKERS) for label in labels)
    has_risky = any(label_has_any(label, RISKY_TEXTURE_MARKERS) for label in labels)
    only_risky = bool(labels) and has_risky and not has_anchor

    if share > max_share:
        return "risky_review", "candidate_share_high"
    if only_risky:
        return "risky_review", "texture_only_without_anchor"
    if has_anchor:
        return "reliable", "has_anchor_label"
    if not has_risky:
        return "reliable", "no_high_risk_texture"
    return "risky_review", "ambiguous_texture_mix"


def annotate(rows: list[dict], max_share: float) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        decision, reason = decision_for(item, max_share=max_share)
        item["reliability"] = decision
        item["reliability_reason"] = reason
        out.append(item)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audio_src(path_value: str, html_path: Path) -> str:
    return Path(os.path.relpath(path_value, html_path.parent)).as_posix()


def count_table(title: str, counter: Counter[str]) -> str:
    body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
    return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"


def write_html(rows: list[dict], out_html: Path, source_path: Path) -> None:
    reliability_counts = Counter(row["reliability"] for row in rows)
    reason_counts = Counter(row["reliability_reason"] for row in rows)
    label_counts = Counter()
    for row in rows:
        label_counts.update(split_cell(row.get("labels", "")))

    trs = []
    for row in sorted(rows, key=lambda r: (r["reliability"] != "reliable", -safe_float(r.get("candidate_rms_share")), r.get("track", ""))):
        mix_src = audio_src(row["stem_mix"], out_html)
        synth_src = audio_src(row["synth_candidate"], out_html)
        residual_src = audio_src(row["residual_context"], out_html)
        cls = html.escape(row["reliability"])
        trs.append(
            "<tr>"
            f"<td><b>{html.escape(row['track'])}</b><br><small>{html.escape(row.get('stems', ''))}</small></td>"
            f"<td class='{cls}'><b>{html.escape(row['reliability'])}</b><br><small>{html.escape(row['reliability_reason'])}</small></td>"
            f"<td>{html.escape(row.get('candidate_rms_share', ''))}</td>"
            f"<td>{html.escape(row.get('candidate_rms_db', ''))}</td>"
            f"<td>{html.escape(row.get('residual_rms_db', ''))}</td>"
            f"<td>{html.escape(row.get('labels', ''))}</td>"
            f"<td><audio controls src='{html.escape(mix_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(synth_src)}'></audio></td>"
            f"<td><audio controls src='{html.escape(residual_src)}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Reliable Synth Candidate Queue</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.reliable {{ background: #dff3e6; }}
.risky_review {{ background: #fff2c8; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Reliable Synth Candidate Queue</h1>
<p class="note">Filtered from <code>{html.escape(str(source_path))}</code>. This is a second-pass reliability filter over auditionable candidates. Risky texture-only labels such as granular/wavetable are kept visible, but moved to review unless supported by stronger synth anchors.</p>
{count_table("Reliability", reliability_counts)}
{count_table("Reliability Reasons", reason_counts)}
{count_table("Labels", label_counts)}
<section>
<h2>Queue</h2>
<table>
<tr><th>Track</th><th>Reliability</th><th>Candidate/Mix RMS</th><th>Candidate dB</th><th>Residual dB</th><th>Labels</th><th>Stem Mix</th><th>Synth Candidate</th><th>Residual Context</th></tr>
{''.join(trs)}
</table>
</section>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auditionable", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_auditionable.csv"))
    parser.add_argument("--max-share", type=float, default=0.78)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_reliable.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_reliable.html"))
    args = parser.parse_args()

    rows = annotate(read_rows(args.auditionable), max_share=args.max_share)
    if not rows:
        raise SystemExit("No auditionable rows to filter.")
    write_csv(rows, args.out_csv)
    write_html(rows, args.out_html, args.auditionable)
    counts = Counter(row["reliability"] for row in rows)
    print(f"rows: {len(rows)}")
    print(f"reliability: {dict(counts.most_common())}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
