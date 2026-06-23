from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


LAYER_ORDER = ["vocals", "drums", "bass", "synth", "guitar_keys", "sample_fx", "noise_fx"]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summarize(likely_rows: list[dict], strong_rows: list[dict]) -> list[dict]:
    tracks = sorted({row["track"] for row in likely_rows} | {row["track"] for row in strong_rows})
    likely_by_track: dict[str, list[dict]] = defaultdict(list)
    strong_by_track: dict[str, list[dict]] = defaultdict(list)
    for row in likely_rows:
        likely_by_track[row["track"]].append(row)
    for row in strong_rows:
        strong_by_track[row["track"]].append(row)

    out = []
    for track in tracks:
        likely = likely_by_track[track]
        strong = strong_by_track[track]
        strong_layers = sorted({row["layer"] for row in strong}, key=lambda x: LAYER_ORDER.index(x))
        stable_likely = [row for row in likely if row["region_type"] == "stable"]
        stable_strong = [row for row in strong if row["region_type"] == "stable"]
        stable_layers = sorted({row["layer"] for row in stable_likely}, key=lambda x: LAYER_ORDER.index(x))
        layer_durations = Counter()
        strong_layer_durations = Counter()
        label_counts = Counter()
        stable_label_counts = Counter()
        for row in likely:
            duration = safe_float(row["duration"])
            layer_durations[row["layer"]] += duration
            label_counts[(row["layer"], row["label"])] += 1
            if row["region_type"] == "stable":
                stable_label_counts[(row["layer"], row["label"])] += 1
        for row in strong:
            strong_layer_durations[row["layer"]] += safe_float(row["duration"])

        def layer_duration_text(counter: Counter) -> str:
            items = [(layer, counter[layer]) for layer in LAYER_ORDER if counter[layer] > 0]
            return "; ".join(f"{layer}:{value:.1f}s" for layer, value in items)

        def top_labels(counter: Counter, limit: int = 8) -> str:
            return "; ".join(f"{layer}/{label}:{count}" for (layer, label), count in counter.most_common(limit))

        flags = []
        if not any(row["layer"] == "synth" for row in likely):
            flags.append("no_synth_candidate")
        elif not any(row["layer"] == "synth" for row in stable_likely):
            flags.append("synth_only_momentary")
        if not any(row["layer"] == "vocals" for row in likely):
            flags.append("no_vocal_candidate")
        if len([row for row in likely if row["region_type"] == "momentary"]) > len(stable_likely) * 2 + 3:
            flags.append("many_momentary_regions")
        if any(row["layer"] == "noise_fx" for row in likely) and not any(row["layer"] == "noise_fx" for row in stable_likely):
            flags.append("noise_fx_momentary")

        out.append(
            {
                "track": track,
                "likely_regions": len(likely),
                "strong_regions": len(strong),
                "stable_likely_regions": len(stable_likely),
                "stable_strong_regions": len(stable_strong),
                "strong_layers": "|".join(strong_layers),
                "stable_layers": "|".join(stable_layers),
                "likely_layer_duration": layer_duration_text(layer_durations),
                "strong_layer_duration": layer_duration_text(strong_layer_durations),
                "top_stable_labels": top_labels(stable_label_counts),
                "top_all_labels": top_labels(label_counts),
                "flags": "|".join(flags),
            }
        )
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict], out_html: Path) -> None:
    layer_counts = Counter()
    flag_counts = Counter()
    for row in rows:
        layer_counts.update([layer for layer in row["stable_layers"].split("|") if layer])
        flag_counts.update([flag for flag in row["flags"].split("|") if flag])

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    trs = []
    for row in rows:
        flags = row["flags"] or "-"
        cls = "flagged" if row["flags"] else ""
        trs.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{row['likely_regions']} / {row['strong_regions']}</td>"
            f"<td>{row['stable_likely_regions']} / {row['stable_strong_regions']}</td>"
            f"<td>{html.escape(row['strong_layers'] or '-')}</td>"
            f"<td>{html.escape(row['stable_layers'] or '-')}</td>"
            f"<td>{html.escape(row['top_stable_labels'] or '-')}</td>"
            f"<td>{html.escape(row['likely_layer_duration'] or '-')}</td>"
            f"<td class='{cls}'>{html.escape(flags)}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Broad Track Overview</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.flagged {{ background: #fff7df; }}
.note {{ color: #444; max-width: 1000px; }}
</style>
<h1>Broad Track Overview</h1>
<p class="note">Clip-level summary from broad layer regions. This is the quickest map of which clips are vocal/synth/bass/drum-heavy and which clips need more review.</p>
{count_table("Stable Layer Coverage", layer_counts)}
{count_table("Flags", flag_counts)}
<h2>Tracks</h2>
<table>
<tr><th>Track</th><th>Likely / Strong Regions</th><th>Stable Likely / Strong</th><th>Strong Layers</th><th>Stable Layers</th><th>Top Stable Labels</th><th>Likely Layer Duration</th><th>Flags</th></tr>
{''.join(trs)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--likely", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_layer_regions_likely.csv"))
    parser.add_argument("--strong", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_layer_regions_strong.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_track_overview.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_track_overview.html"))
    args = parser.parse_args()

    rows = summarize(read_rows(args.likely), read_rows(args.strong))
    write_csv(rows, args.out_csv)
    write_html(rows, args.out_html)
    print(f"tracks: {len(rows)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
