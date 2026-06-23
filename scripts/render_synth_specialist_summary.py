from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


ACTIVE_STRENGTHS = {"medium", "strong"}
STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def is_active(row: dict) -> bool:
    return row.get("strength") in ACTIVE_STRENGTHS


def sort_key(row: dict) -> tuple[str, float, str]:
    return row["track"], float(row["start"]), row["stem"]


def stem_rank(stem: str) -> int:
    return STEM_ORDER.index(stem) if stem in STEM_ORDER else len(STEM_ORDER)


def summarize_tracks(rows: list[dict]) -> list[dict]:
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)

    out = []
    for track, track_rows in sorted(by_track.items()):
        active = [row for row in track_rows if is_active(row)]
        labels = Counter(row["synth_label_top"] for row in active)
        stems = Counter(row["stem"] for row in active)
        strengths = Counter(row["strength"] for row in track_rows)
        out.append(
            {
                "track": track,
                "segments": len(track_rows),
                "active_synth_segments": len(active),
                "active_ratio": round(len(active) / max(len(track_rows), 1), 4),
                "top_synth_labels": "|".join(f"{label}:{count}" for label, count in labels.most_common(6)) or "-",
                "active_stems": "|".join(f"{stem}:{count}" for stem, count in sorted(stems.items(), key=lambda item: (stem_rank(item[0]), -item[1]))) or "-",
                "strengths": "|".join(f"{label}:{count}" for label, count in strengths.most_common()) or "-",
            }
        )
    return out


def summarize_regions(rows: list[dict]) -> list[dict]:
    active_rows = [row for row in rows if is_active(row)]
    by_track_stem: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in active_rows:
        by_track_stem[(row["track"], row["stem"])].append(row)

    regions = []
    for (track, stem), stem_rows in sorted(by_track_stem.items(), key=lambda item: (item[0][0], stem_rank(item[0][1]))):
        current = None
        for row in sorted(stem_rows, key=lambda item: float(item["start"])):
            signature = (row["synth_label_top"], row["strength"])
            if (
                current
                and current["signature"] == signature
                and abs(float(current["end"]) - float(row["start"])) < 1e-6
            ):
                current["end"] = row["end"]
                current["scores"].append(float(row["synth_label_conf"]))
                current["segments"].append(f"{row['start']}-{row['end']}")
                continue
            if current:
                regions.append(current)
            current = {
                "track": track,
                "stem": stem,
                "start": row["start"],
                "end": row["end"],
                "synth_label": row["synth_label_top"],
                "family": row["synth_family_top"],
                "strength": row["strength"],
                "signature": signature,
                "scores": [float(row["synth_label_conf"])],
                "segments": [f"{row['start']}-{row['end']}"],
            }
        if current:
            regions.append(current)

    out = []
    for region in regions:
        out.append(
            {
                "track": region["track"],
                "stem": region["stem"],
                "start": region["start"],
                "end": region["end"],
                "synth_label": region["synth_label"],
                "family": region["family"],
                "strength": region["strength"],
                "mean_confidence": round(sum(region["scores"]) / len(region["scores"]), 6),
                "segments": "|".join(region["segments"]),
            }
        )
    return out


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def write_html(track_rows: list[dict], region_rows: list[dict], out_html: Path) -> None:
    label_counts = Counter()
    stem_counts = Counter()
    for row in region_rows:
        label_counts[row["synth_label"]] += 1
        stem_counts[row["stem"]] += 1

    overview = table(
        ["Metric", "Value"],
        [
            ["Tracks", len(track_rows)],
            ["Synth-active regions", len(region_rows)],
            ["Top labels", ", ".join(f"{label} ({count})" for label, count in label_counts.most_common(8)) or "-"],
            ["Active stems", ", ".join(f"{stem} ({count})" for stem, count in sorted(stem_counts.items(), key=lambda item: stem_rank(item[0]))) or "-"],
        ],
    )
    tracks = table(
        ["Track", "Active Ratio", "Top Synth Labels", "Active Stems", "Strength Cells"],
        [
            [
                row["track"],
                row["active_ratio"],
                row["top_synth_labels"],
                row["active_stems"],
                row["strengths"],
            ]
            for row in track_rows
        ],
    )
    regions = table(
        ["Track", "Stem", "Time", "Synth", "Family", "Strength", "Mean Confidence", "Segments"],
        [
            [
                row["track"],
                row["stem"],
                f"{row['start']}-{row['end']}s",
                row["synth_label"],
                row["family"],
                row["strength"],
                row["mean_confidence"],
                row["segments"],
            ]
            for row in region_rows
        ],
    )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Specialist Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Synth Specialist Summary</h1>
<p class="note">Compact view of synth-active regions from the focused synth specialist. Use this before opening the full row-level specialist table.</p>
<h2>Overview</h2>
{overview}
<h2>Track Summary</h2>
{tracks}
<h2>Synth Regions</h2>
{regions}
<p class="note">Caution: regions are grouped model hypotheses. They are not exact VST, preset, oscillator, or clean source-separation truth.</p>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist.csv"))
    parser.add_argument("--out-track-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist_track_summary.csv"))
    parser.add_argument("--out-region-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist_regions.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist_summary.html"))
    args = parser.parse_args()

    rows = read_rows(args.input)
    track_rows = summarize_tracks(rows)
    region_rows = summarize_regions(rows)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_csv(track_rows, args.out_track_csv)
    write_csv(region_rows, args.out_region_csv)
    write_html(track_rows, region_rows, args.out_html)
    print(f"wrote: {args.out_track_csv}")
    print(f"wrote: {args.out_region_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
