from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def compact_labels(value: str, limit: int) -> str:
    labels = [item for item in str(value or "").split("|") if item]
    return "<br>".join(html.escape(label) for label in labels[:limit]) if labels else "-"


def stem_order(rows: list[dict]) -> list[str]:
    found = {row["stem"] for row in rows}
    ordered = [stem for stem in DEFAULT_STEM_ORDER if stem in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def confidence_class(top_source_kinds: str, detected_source_kinds: str) -> tuple[str, str]:
    detected = [item for item in str(detected_source_kinds or "").split("|") if item]
    if not detected:
        return "empty", "empty"
    first = str(top_source_kinds or "").split(";", 1)[0].strip()
    try:
        score_text = first.split(":", 1)[1].split("/", 2)
        score = float(score_text[0])
        threshold = float(score_text[1])
    except (IndexError, ValueError):
        return "medium", "medium"
    if threshold <= 0:
        return "medium", "medium"
    ratio = score / threshold
    if ratio >= 1.35:
        return "strong", "strong"
    if ratio >= 1.08:
        return "medium", "medium"
    return "weak", "weak"


def write_html(rows: list[dict], out_html: Path, label_limit: int) -> None:
    stems = stem_order(rows)
    by_track: dict[str, dict[tuple[str, str], dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    label_counts = Counter()
    stem_label_counts: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        track = row["track"]
        key = (row["start"], row["end"])
        by_track[track][key][row["stem"]] = row
        labels = [label for label in row["detected_source_kinds"].split("|") if label]
        label_counts.update(labels)
        stem_label_counts[row["stem"]].update(labels)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common(18))
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for track, segments in sorted(by_track.items()):
        segment_rows = []
        for start, end in sorted(segments.keys(), key=lambda item: float(item[0])):
            cells = []
            for stem in stems:
                row = segments[(start, end)].get(stem)
                if not row:
                    cells.append("<td>-</td>")
                    continue
                labels = compact_labels(row["detected_source_kinds"], label_limit)
                groups = compact_labels(row.get("detected_source_groups", ""), label_limit)
                top = html.escape(row["top_source_kinds"])
                cls, badge = confidence_class(row["top_source_kinds"], row["detected_source_kinds"])
                cells.append(f"<td class='{cls}'><b>{badge}</b><br><small>{groups}</small><br>{labels}<details><summary>scores</summary>{top}</details></td>")
            segment_rows.append(f"<tr><td>{start}-{end}s</td>{''.join(cells)}</tr>")
        header_cells = "".join(f"<th>{html.escape(stem.title())} stem</th>" for stem in stems)
        sections.append(
            f"""<section>
<h2>{html.escape(track)}</h2>
<table>
<tr><th>Time</th>{header_cells}</tr>
{''.join(segment_rows)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Stem Source Timeline Matrix</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
td {{ min-width: 160px; }}
small {{ color: #555; font-weight: bold; }}
.strong {{ background: #e8f7ed; }}
.medium {{ background: #fff7df; }}
.weak {{ background: #ffe9dc; }}
.empty {{ background: #f4f4f4; color: #777; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
details {{ margin-top: 6px; color: #444; max-width: 360px; }}
summary {{ cursor: pointer; color: #333; }}
</style>
<h1>Demucs Stem Source Timeline Matrix</h1>
<p>Each mixed clip is split into vocals, drums, bass, and other stems. Labels are stem-aware source-kind estimates per 2-second segment.</p>
<p><b>Cell confidence:</b> strong = top candidate clearly above threshold, medium = usable candidate, weak = borderline, empty = no stem-valid candidate.</p>
{count_table("All Detected Source Kinds", label_counts)}
{''.join(count_table(f"{stem} Stem", stem_label_counts[stem]) for stem in stems)}
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_full/stem_source_kind_strict.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_full/stem_timeline_matrix.html"))
    parser.add_argument("--label-limit", type=int, default=4)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    write_html(rows, args.out_html, args.label_limit)
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
