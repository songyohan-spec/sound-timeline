from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


SYNTH_LABELS = {
    "synth_pad_or_wash",
    "supersaw_or_bright_synth_stack",
    "synth_pluck_or_bell",
    "arpeggio_or_sequence_synth",
    "digital_synth_lead",
    "bitcrushed_or_aliasing_synth",
    "fuzzy_distorted_synth",
    "wavetable_noise_synth",
    "granular_or_resampled_synth",
    "synth_bass",
    "sidechained_bass_pulse",
    "vocal_synth_hybrid",
    "formant_or_vocoder_vocal",
}

BASS_SYNTH_LABELS = {"synth_bass", "sidechained_bass_pulse"}

SYNTH_STEM_PRIORITY = {
    "other": 1.12,
    "piano": 1.08,
    "bass": 1.05,
    "guitar": 1.00,
    "vocals": 0.95,
    "drums": 0.72,
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def split_pipe(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def parse_top_entries(value: str) -> list[dict]:
    entries = []
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
            threshold = float(parts[1])
        except ValueError:
            continue
        entries.append(
            {
                "label": label.strip(),
                "score": score,
                "threshold": threshold,
                "status": parts[2].strip(),
            }
        )
    return entries


def is_synth_label(label: str) -> bool:
    return label in SYNTH_LABELS or "synth" in label or "wavetable" in label or "granular" in label


def synth_entries(row: dict, min_score: float) -> list[dict]:
    out = []
    detected = set(split_pipe(row.get("detected_source_kinds", "")))
    priority = SYNTH_STEM_PRIORITY.get(row.get("stem", ""), 1.0)
    for entry in parse_top_entries(row.get("top_source_kinds", "")):
        label = entry["label"]
        if not is_synth_label(label):
            continue
        if entry["score"] < min_score and label not in detected:
            continue
        adjusted = entry["score"] * priority
        if label in detected:
            adjusted *= 1.18
        out.append({**entry, "adjusted_score": adjusted, "stem": row.get("stem", ""), "detected": label in detected})
    return sorted(out, key=lambda item: item["adjusted_score"], reverse=True)


def segment_key(row: dict) -> tuple[str, str, str]:
    return row["track"], row["start"], row["end"]


def classify_strength(score: float) -> str:
    if score >= 0.22:
        return "strong"
    if score >= 0.12:
        return "medium"
    return "weak"


def compact(entries: list[dict], limit: int = 5) -> str:
    cells = []
    for entry in entries[:limit]:
        detected = ", detected" if entry["detected"] else ""
        cells.append(f"{entry['label']}@{entry['stem']} ({entry['score']:.3f}{detected})")
    return "|".join(cells)


def build_rows(rows: list[dict], min_score: float) -> list[dict]:
    by_segment: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        entries = synth_entries(row, min_score)
        if entries:
            by_segment[segment_key(row)].extend(entries)

    out = []
    for (track, start, end), entries in sorted(by_segment.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        entries = sorted(entries, key=lambda item: item["adjusted_score"], reverse=True)
        best = entries[0]
        non_bass_entries = [entry for entry in entries if entry["label"] not in BASS_SYNTH_LABELS]
        best_non_bass = non_bass_entries[0] if non_bass_entries else None
        labels = Counter(entry["label"] for entry in entries)
        stems = Counter(entry["stem"] for entry in entries)
        out.append(
            {
                "track": track,
                "start": start,
                "end": end,
                "primary_synth": best["label"],
                "primary_stem": best["stem"],
                "primary_score": round(best["score"], 6),
                "adjusted_score": round(best["adjusted_score"], 6),
                "strength": classify_strength(best["adjusted_score"]),
                "primary_non_bass_synth": best_non_bass["label"] if best_non_bass else "-",
                "non_bass_stem": best_non_bass["stem"] if best_non_bass else "-",
                "non_bass_score": round(best_non_bass["score"], 6) if best_non_bass else 0.0,
                "non_bass_adjusted_score": round(best_non_bass["adjusted_score"], 6) if best_non_bass else 0.0,
                "non_bass_strength": classify_strength(best_non_bass["adjusted_score"]) if best_non_bass else "none",
                "candidate_synths": compact(entries),
                "candidate_non_bass_synths": compact(non_bass_entries),
                "synth_label_mentions": "|".join(label for label, _ in labels.most_common(6)),
                "stem_mentions": "|".join(f"{stem}:{count}" for stem, count in stems.most_common()),
            }
        )
    return out


def summarize_regions(rows: list[dict]) -> list[dict]:
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)

    regions = []
    for track, track_rows in sorted(by_track.items()):
        current = None
        for row in sorted(track_rows, key=lambda item: float(item["start"])):
            signature = (row["primary_synth"], row["primary_stem"], row["strength"])
            if current and current["signature"] == signature and abs(float(current["end"]) - float(row["start"])) < 1e-6:
                current["end"] = row["end"]
                current["segments"].append(f"{row['start']}-{row['end']}")
                current["scores"].append(float(row["adjusted_score"]))
                continue
            if current:
                regions.append(current)
            current = {
                "track": track,
                "start": row["start"],
                "end": row["end"],
                "primary_synth": row["primary_synth"],
                "primary_stem": row["primary_stem"],
                "strength": row["strength"],
                "signature": signature,
                "segments": [f"{row['start']}-{row['end']}"],
                "scores": [float(row["adjusted_score"])],
            }
        if current:
            regions.append(current)

    out = []
    for region in regions:
        out.append(
            {
                "track": region["track"],
                "start": region["start"],
                "end": region["end"],
                "primary_synth": region["primary_synth"],
                "primary_stem": region["primary_stem"],
                "strength": region["strength"],
                "mean_adjusted_score": round(sum(region["scores"]) / len(region["scores"]), 6),
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


def write_html(segment_rows: list[dict], region_rows: list[dict], out_html: Path) -> None:
    label_counts = Counter(row["primary_synth"] for row in segment_rows)
    non_bass_label_counts = Counter(row["primary_non_bass_synth"] for row in segment_rows if row["primary_non_bass_synth"] != "-")
    stem_counts = Counter(row["primary_stem"] for row in segment_rows)
    strength_counts = Counter(row["strength"] for row in segment_rows)

    label_table = table(["Synth Label", "Segments"], [[label, count] for label, count in label_counts.most_common()])
    non_bass_label_table = table(["Non-Bass Synth Label", "Segments"], [[label, count] for label, count in non_bass_label_counts.most_common()])
    stem_table = table(["Primary Stem", "Segments"], [[label, count] for label, count in stem_counts.most_common()])
    strength_table = table(["Strength", "Segments"], [[label, count] for label, count in strength_counts.most_common()])
    region_table = table(
        ["Track", "Time", "Primary Synth", "Stem", "Strength", "Mean Score", "Segments"],
        [
            [
                row["track"],
                f"{row['start']}-{row['end']}s",
                row["primary_synth"],
                row["primary_stem"],
                row["strength"],
                row["mean_adjusted_score"],
                row["segments"],
            ]
            for row in region_rows
        ],
    )
    detail_table = table(
        ["Track", "Time", "Primary Synth", "Stem", "Non-Bass Synth", "Non-Bass Stem", "Score", "Strength", "Candidates"],
        [
            [
                row["track"],
                f"{row['start']}-{row['end']}s",
                row["primary_synth"],
                row["primary_stem"],
                row["primary_non_bass_synth"],
                row["non_bass_stem"],
                row["adjusted_score"],
                row["strength"],
                row["candidate_non_bass_synths"] or row["candidate_synths"],
            ]
            for row in segment_rows
        ],
    )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Synth Focus</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Demucs Synth Focus</h1>
<p class="note">This is a synth lens over Demucs stems. It does not assume a separate synth stem exists. Instead, it pulls synth-like candidates from vocals, bass, guitar, piano, and other stems, because commercial synth layers often leak into those stems.</p>
<div class="grid">
<section><h2>Primary Synth Labels</h2>{label_table}</section>
<section><h2>Non-Bass Synth Labels</h2>{non_bass_label_table}</section>
<section><h2>Where They Appear</h2>{stem_table}</section>
<section><h2>Strength</h2>{strength_table}</section>
</div>
<h2>Synth Regions</h2>
{region_table}
<h2>Segment Detail</h2>
{detail_table}
<p class="note">Caution: synth labels are source-kind hypotheses, not oscillator/preset identification. Weak detections mainly mean the current model needs stronger synth-specific training examples.</p>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_focus.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_focus.html"))
    parser.add_argument("--min-score", type=float, default=0.045)
    args = parser.parse_args()

    segment_rows = build_rows(read_rows(args.input), args.min_score)
    if not segment_rows:
        raise SystemExit("No synth-like candidates found. Try lowering --min-score.")
    region_rows = summarize_regions(segment_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(segment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(segment_rows)
    write_html(segment_rows, region_rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
