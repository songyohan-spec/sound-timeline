from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path


SYNTH_LAYERS = {"synth", "vocals", "bass"}
DETAIL_PREFIX = "synth_detail:"


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_cell(value: str) -> list[dict]:
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
        items.append({"strength": strength, "label": label, "stem": stem, "score": safe_float(score)})
    return items


def two_s_synth_labels(row: dict) -> set[str]:
    labels = set()
    for layer in SYNTH_LAYERS:
        for item in parse_cell(row.get(layer, "")):
            label = item["label"]
            if label.startswith(DETAIL_PREFIX) or "synth" in label or "vocoder" in label or "formant" in label:
                labels.add(label.replace(DETAIL_PREFIX, ""))
    return labels


def four_s_rows_by_track_stem(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        out[(row["track"], row["stem"])].append(row)
    for key in out:
        out[key].sort(key=lambda row: safe_float(row["start"]))
    return out


def overlaps(start: float, end: float, row: dict) -> bool:
    return safe_float(row["start"]) <= start + 0.01 and safe_float(row["end"]) >= end - 0.01


def four_s_candidates(rows: list[dict], track: str, start: float, end: float) -> list[dict]:
    candidates = []
    for (row_track, _stem), stem_rows in rows.items():
        if row_track != track:
            continue
        for row in stem_rows:
            if overlaps(start, end, row) and row.get("final_label") != "ambiguous":
                candidates.append(row)
    candidates.sort(key=lambda row: (row["decision"] == "use_pseudo_label", safe_float(row["specialist_conf"])), reverse=True)
    return candidates


def classify(two_labels: set[str], candidates: list[dict]) -> tuple[str, str]:
    four_labels = {row["final_label"] for row in candidates if row.get("final_label") != "ambiguous"}
    if not four_labels:
        return "no_4s_support", ""
    if two_labels & four_labels:
        return "confirmed_by_4s", "|".join(sorted(two_labels & four_labels))
    if two_labels and four_labels:
        return "evolving_or_label_shift", "|".join(sorted(four_labels))
    return "4s_only_synth_context", "|".join(sorted(four_labels))


def write_html(rows: list[dict], out_html: Path) -> None:
    status_counts = Counter(row["multiscale_status"] for row in rows)
    label_counts = Counter()
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
        label_counts.update([label for label in row["four_s_labels"].split("|") if label])

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    sections = []
    for track, track_rows in sorted(by_track.items()):
        trs = []
        for row in sorted(track_rows, key=lambda r: safe_float(r["start"])):
            cls = html.escape(row["multiscale_status"])
            trs.append(
                "<tr>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td>{html.escape(row['two_s_synth_labels'] or '-')}</td>"
                f"<td>{html.escape(row['four_s_labels'] or '-')}</td>"
                f"<td class='{cls}'>{html.escape(row['multiscale_status'])}</td>"
                f"<td>{html.escape(row['four_s_support_detail'])}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(track)}</h2>
<table>
<tr><th>Time</th><th>2s Synth Hints</th><th>4s Synth Context</th><th>Status</th><th>4s Detail</th></tr>
{''.join(trs)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Multi-Scale Synth Overlay</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.confirmed_by_4s {{ background: #dff3e6; }}
.evolving_or_label_shift {{ background: #fff2c8; }}
[class="4s_only_synth_context"], .no_4s_support {{ background: #f3f3f3; }}
.note {{ color: #444; max-width: 1000px; }}
</style>
<h1>Multi-Scale Synth Overlay</h1>
<p class="note">Compares 2-second synth hints with 4-second parent-window synth evidence. This helps separate real evolving synth textures from 2-second label flicker.</p>
{count_table("Multi-Scale Status", status_counts)}
{count_table("4s Synth Labels", label_counts)}
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeline-2s", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_multilayer_timeline.csv"))
    parser.add_argument("--synth-4s", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v4_4s_hop2.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/multiscale_synth_overlay.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/multiscale_synth_overlay.html"))
    args = parser.parse_args()

    four_by = four_s_rows_by_track_stem(read_rows(args.synth_4s))
    out_rows = []
    for row in read_rows(args.timeline_2s):
        start = safe_float(row["start"])
        end = safe_float(row["end"])
        two_labels = two_s_synth_labels(row)
        candidates = four_s_candidates(four_by, row["track"], start, end)
        status, labels = classify(two_labels, candidates)
        detail = "; ".join(
            f"{cand['stem']}/{cand['final_label']}:{cand['specialist_conf']}:{cand['decision']}"
            for cand in candidates[:8]
        )
        out = {
            "track": row["track"],
            "start": row["start"],
            "end": row["end"],
            "two_s_synth_labels": "|".join(sorted(two_labels)),
            "four_s_labels": labels,
            "multiscale_status": status,
            "four_s_support_detail": detail,
        }
        out_rows.append(out)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    write_html(out_rows, args.out_html)
    print(f"rows: {len(out_rows)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
