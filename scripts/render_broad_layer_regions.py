from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


LAYER_ORDER = ["vocals", "drums", "bass", "synth", "guitar_keys", "sample_fx", "noise_fx"]
STRENGTH_VALUE = {"strong": 3, "likely": 2, "possible": 1}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_layer_cell(value: str) -> list[dict]:
    items = []
    for raw in str(value or "").split("|"):
        raw = raw.strip()
        if not raw:
            continue
        try:
            strength, rest = raw.split(":", 1)
            label_stem, score = rest.rsplit(":", 1)
            label, stem = label_stem.rsplit("@", 1)
        except ValueError:
            continue
        items.append(
            {
                "strength": strength,
                "label": label,
                "stem": stem,
                "score": safe_float(score),
            }
        )
    items.sort(key=lambda item: (STRENGTH_VALUE.get(item["strength"], 0), item["score"]), reverse=True)
    return items


def is_synth_like_label(label: str) -> bool:
    value = label.lower()
    return "synth" in value or "vocoder" in value or "formant" in value


def best_item(row: dict, layer: str, min_strength: str, use_synth_context: bool = False) -> dict | None:
    min_value = STRENGTH_VALUE[min_strength]
    for item in parse_layer_cell(row.get(layer, "")):
        if use_synth_context and row.get("synth_4s_status") == "no_4s_support" and is_synth_like_label(item["label"]):
            continue
        if STRENGTH_VALUE.get(item["strength"], 0) >= min_value:
            return item
    return None


def merge_regions(rows: list[dict], min_strength: str, use_synth_context: bool = False) -> list[dict]:
    rows = sorted(rows, key=lambda row: (row["track"], safe_float(row["start"])))
    active: dict[tuple[str, str, str], dict] = {}
    finished = []

    for row in rows:
        track = row["track"]
        start = safe_float(row["start"])
        end = safe_float(row["end"])
        present_keys = set()
        for layer in LAYER_ORDER:
            item = best_item(row, layer, min_strength, use_synth_context=use_synth_context)
            if not item:
                continue
            key = (track, layer, item["label"])
            present_keys.add(key)
            current = active.get(key)
            if current and abs(safe_float(current["end"]) - start) <= 0.05:
                current["end"] = row["end"]
                current["segments"] += 1
                current["max_score"] = max(current["max_score"], item["score"])
                if STRENGTH_VALUE[item["strength"]] > STRENGTH_VALUE[current["max_strength"]]:
                    current["max_strength"] = item["strength"]
                current["stems"].add(item["stem"])
            else:
                if current:
                    finished.append(current)
                active[key] = {
                    "track": track,
                    "layer": layer,
                    "label": item["label"],
                    "start": row["start"],
                    "end": row["end"],
                    "segments": 1,
                    "max_strength": item["strength"],
                    "max_score": item["score"],
                    "stems": {item["stem"]},
                }
        for key in list(active):
            if key[0] == track and key not in present_keys and safe_float(active[key]["end"]) <= start + 0.05:
                finished.append(active.pop(key))

    finished.extend(active.values())
    out = []
    for region in finished:
        duration = safe_float(region["end"]) - safe_float(region["start"])
        row = dict(region)
        row["duration"] = round(duration, 4)
        row["stems"] = "|".join(sorted(region["stems"]))
        row["max_score"] = round(region["max_score"], 6)
        row["region_type"] = "stable" if region["segments"] >= 2 else "momentary"
        out.append(row)
    out.sort(key=lambda r: (r["track"], safe_float(r["start"]), r["layer"], -STRENGTH_VALUE.get(r["max_strength"], 0)))
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["track", "layer", "label", "start", "end", "duration", "segments", "region_type", "max_strength", "max_score", "stems"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row[key] for key in fieldnames} for row in rows])


def write_html(regions: list[dict], out_html: Path, min_strength: str) -> None:
    by_layer = Counter(row["layer"] for row in regions)
    by_type = Counter(row["region_type"] for row in regions)
    by_strength = Counter(row["max_strength"] for row in regions)
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in regions:
        by_track[row["track"]].append(row)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for track, rows in sorted(by_track.items()):
        trs = []
        for row in rows:
            cls = html.escape(row["max_strength"])
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s<br><small>{row['duration']}s</small></td>"
                f"<td>{html.escape(row['layer'])}</td>"
                f"<td><b>{html.escape(row['label'])}</b><br><small>{html.escape(row['stems'])}</small></td>"
                f"<td class='{cls}'>{html.escape(row['max_strength'])}</td>"
                f"<td>{row['region_type']}</td>"
                f"<td>{row['segments']}</td>"
                f"<td>{row['max_score']}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(track)}</h2>
<table>
<tr><th>Time</th><th>Layer</th><th>Candidate</th><th>Strength</th><th>Type</th><th>Segments</th><th>Score</th></tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Broad Layer Regions</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.strong {{ background: #dff3e6; }}
.likely {{ background: #fff2c8; }}
.possible {{ background: #f3f3f3; color: #444; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1000px; }}
</style>
<h1>Broad Layer Regions</h1>
<p class="note">Contiguous regions derived from the broad multi-layer timeline. Min strength: <b>{html.escape(min_strength)}</b>. Stable means the same layer/label lasted for at least two 2-second segments.</p>
<p>Regions: {len(regions)}</p>
{count_table("Layers", by_layer)}
{count_table("Region Type", by_type)}
{count_table("Strength", by_strength)}
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline.csv"))
    parser.add_argument("--min-strength", choices=["possible", "likely", "strong"], default="likely")
    parser.add_argument("--use-synth-context", action="store_true")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_layer_regions_likely.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_layer_regions_likely.html"))
    args = parser.parse_args()

    regions = merge_regions(read_rows(args.input), args.min_strength, use_synth_context=args.use_synth_context)
    write_csv(regions, args.out_csv)
    write_html(regions, args.out_html, args.min_strength)
    print(f"regions: {len(regions)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
