from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def split_pipe(value: str) -> list[str]:
    return [item for item in str(value or "").split("|") if item]


def top_items(counter: Counter[str], limit: int) -> str:
    return "|".join(label for label, _ in counter.most_common(limit))


def confidence(top_source_kinds: str, detected_source_kinds: str) -> str:
    if not split_pipe(detected_source_kinds):
        return "empty"
    first = str(top_source_kinds or "").split(";", 1)[0].strip()
    try:
        score_text = first.split(":", 1)[1].split("/", 2)
        ratio = float(score_text[0]) / max(float(score_text[1]), 1e-8)
    except (IndexError, ValueError):
        return "medium"
    if ratio >= 1.35:
        return "strong"
    if ratio >= 1.08:
        return "medium"
    return "weak"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def stem_order(rows: list[dict]) -> list[str]:
    found = {row["stem"] for row in rows}
    ordered = [stem for stem in DEFAULT_STEM_ORDER if stem in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def summarize(rows: list[dict], top_k: int) -> tuple[list[dict], list[str]]:
    stems = stem_order(rows)
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)

    summaries = []
    for track, track_rows in sorted(by_track.items()):
        out = {"track": track, "segments": len(track_rows)}
        confidence_counts = Counter()
        for stem in stems:
            stem_rows = [row for row in track_rows if row["stem"] == stem]
            counter = Counter()
            group_counter = Counter()
            for row in stem_rows:
                counter.update(split_pipe(row["detected_source_kinds"]))
                group_counter.update(split_pipe(row.get("detected_source_groups", "")))
                confidence_counts[confidence(row["top_source_kinds"], row["detected_source_kinds"])] += 1
            out[f"{stem}_top"] = top_items(counter, top_k) or "-"
            out[f"{stem}_groups"] = top_items(group_counter, top_k) or "-"
            out[f"{stem}_active_segments"] = sum(1 for row in stem_rows if split_pipe(row["detected_source_kinds"]))
            out[f"{stem}_total_segments"] = len(stem_rows)
        out["strong_cells"] = confidence_counts["strong"]
        out["medium_cells"] = confidence_counts["medium"]
        out["weak_cells"] = confidence_counts["weak"]
        out["empty_cells"] = confidence_counts["empty"]
        summaries.append(out)
    return summaries, stems


def write_html(rows: list[dict], stems: list[str], out_html: Path) -> None:
    body = []
    for row in rows:
        stem_cells = ""
        for stem in stems:
            stem_cells += (
                f"<td><b>{html.escape(row.get(f'{stem}_groups', '-'))}</b><br>"
                f"{html.escape(row.get(f'{stem}_top', '-'))}<br>"
                f"<small>{row.get(f'{stem}_active_segments', 0)}/{row.get(f'{stem}_total_segments', 0)} active</small></td>"
            )
        body.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{row['segments']}</td>"
            f"{stem_cells}"
            f"<td>strong {row['strong_cells']}<br>medium {row['medium_cells']}<br>weak {row['weak_cells']}<br>empty {row['empty_cells']}</td>"
            "</tr>"
        )
    stem_headers = "".join(f"<th>{html.escape(stem.title())}</th>" for stem in stems)

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Collection Summary</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
small {{ color: #555; }}
</style>
<h1>Demucs Collection Summary</h1>
<p>Track-level summary of stem-aware source-kind candidates. Use this before opening the full timeline matrix.</p>
<p><b>Active</b> means a 2-second stem segment produced at least one stem-valid source-kind above the current threshold. Inactive can mean silence, weak separated signal, or candidates filtered out by stem-aware gating.</p>
<table>
<tr><th>Track</th><th>Rows</th>{stem_headers}<th>Confidence Cells</th></tr>
{''.join(body)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_full/stem_source_kind.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_full/collection_summary.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_full/collection_summary.html"))
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    rows, stems = summarize(read_rows(args.input), args.top_k)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, stems, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
