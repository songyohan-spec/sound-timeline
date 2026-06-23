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


def norm_time(value: str) -> str:
    return str(float(value))


def time_key(row: dict) -> tuple[str, str, str]:
    return row["track"], norm_time(row["start"]), norm_time(row["end"])


def fmt_cell(value: str, suspect: bool = False) -> str:
    if not value:
        return "<span class='empty'>-</span>"
    parts = []
    for raw in value.split("|"):
        try:
            strength, rest = raw.split(":", 1)
            label_stem, score = rest.rsplit(":", 1)
            label, stem = label_stem.rsplit("@", 1)
        except ValueError:
            parts.append(html.escape(raw))
            continue
        cls = html.escape(strength)
        extra = " suspect" if suspect and ("synth" in label or "vocoder" in label or "formant" in label) else ""
        parts.append(
            f"<div class='{cls}{extra}'><b>{html.escape(strength)}</b> "
            f"{html.escape(label)} <small>{html.escape(stem)} {html.escape(score)}</small></div>"
        )
    return "".join(parts)


def merge_rows(broad_rows: list[dict], overlay_rows: list[dict]) -> list[dict]:
    overlay_by_time = {time_key(row): row for row in overlay_rows}
    out = []
    for row in broad_rows:
        merged = dict(row)
        overlay = overlay_by_time.get(time_key(row))
        if overlay:
            merged["synth_4s_status"] = overlay.get("multiscale_status", "")
            merged["synth_4s_labels"] = overlay.get("four_s_labels", "")
            merged["synth_2s_labels"] = overlay.get("two_s_synth_labels", "")
        else:
            merged["synth_4s_status"] = "not_checked"
            merged["synth_4s_labels"] = ""
            merged["synth_2s_labels"] = ""
        out.append(merged)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict], path: Path) -> None:
    status_counts = Counter(row["synth_4s_status"] for row in rows)
    label_counts = Counter()
    for row in rows:
        label_counts.update([label for label in row["synth_4s_labels"].split("|") if label])

    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for track, track_rows in sorted(by_track.items()):
        trs = []
        for row in sorted(track_rows, key=lambda r: safe_float(r["start"])):
            status = row["synth_4s_status"]
            suspect = status == "no_4s_support"
            status_cls = html.escape(status)
            cells = "".join(f"<td>{fmt_cell(row[layer], suspect=suspect)}</td>" for layer in LAYER_ORDER)
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td class='{status_cls}'><b>{html.escape(status)}</b><br>"
                f"<small>4s: {html.escape(row['synth_4s_labels'] or '-')}</small><br>"
                f"<small>2s: {html.escape(row['synth_2s_labels'] or '-')}</small></td>"
                f"<td><b>strong</b>: {html.escape(row['strong_layers'] or '-')}<br>"
                f"<b>likely</b>: {html.escape(row['likely_layers'] or '-')}<br>"
                f"<b>possible</b>: {html.escape(row['possible_layers'] or '-')}</td>"
                f"{cells}"
                "</tr>"
            )
        header = "".join(f"<th>{html.escape(layer)}</th>" for layer in LAYER_ORDER)
        sections.append(
            f"<section><h2>{html.escape(track)}</h2><table>"
            f"<tr><th>Time</th><th>Synth 4s Context</th><th>Active</th>{header}</tr>{''.join(trs)}</table></section>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Broad Multi-Layer Timeline + 4s Synth Context</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
td {{ min-width: 145px; }}
.strong {{ background: #dff3e6; padding: 3px; margin: 2px 0; }}
.likely {{ background: #fff2c8; padding: 3px; margin: 2px 0; }}
.possible {{ background: #f3f3f3; padding: 3px; margin: 2px 0; color: #444; }}
.suspect {{ outline: 2px solid #d98080; }}
.confirmed_by_4s {{ background: #dff3e6; }}
.evolving_or_label_shift {{ background: #fff2c8; }}
[class="4s_only_synth_context"] {{ background: #e8eefc; }}
.no_4s_support {{ background: #f7dede; }}
.not_checked {{ background: #f3f3f3; color: #555; }}
.empty {{ color: #888; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Broad Multi-Layer Timeline + 4s Synth Context</h1>
<p class="note">This is the main exploratory layer timeline with a 4-second synth-context cross-check. Red-outlined synth/vocal-synth candidates have no 4-second support and should be treated as short-window flicker unless another layer cue is strong.</p>
{count_table("Synth 4s Status", status_counts)}
{count_table("4s Synth Labels", label_counts)}
{''.join(sections)}
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline.csv"))
    parser.add_argument("--multiscale", type=Path, default=Path("outputs/demucs_stems_6s_full/multiscale_synth_overlay.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline_multiscale.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline_multiscale.html"))
    args = parser.parse_args()

    rows = merge_rows(read_rows(args.broad), read_rows(args.multiscale))
    if not rows:
        raise SystemExit("No rows to render.")
    write_csv(rows, args.out_csv)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
