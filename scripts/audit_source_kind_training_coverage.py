from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def split_pipe(value: str) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def count_source_files(root: Path, labels: list[str]) -> dict[str, int]:
    counts = {}
    for label in labels:
        folder = root / label
        if not folder.exists():
            counts[label] = 0
            continue
        counts[label] = len([path for path in folder.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS])
    return counts


def count_dataset_examples(metadata: Path, labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    if not metadata.exists():
        return counts
    with metadata.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            row_labels = []
            if "label" in row:
                row_labels.append(str(row["label"]))
            row_labels.extend(split_pipe(row.get("labels", "")))
            for label in labels:
                if label in row_labels:
                    counts[label] += 1
    return counts


def status_for(source_files: int, examples: int, mentions: int) -> str:
    if source_files == 0 and examples == 0 and mentions > 0:
        return "missing_training_data"
    if source_files < 3 and mentions >= 10:
        return "under_supported"
    if examples == 0 and mentions > 0:
        return "not_trained_yet"
    if mentions == 0 and (source_files > 0 or examples > 0):
        return "trained_but_not_seen"
    return "ok"


def write_html(rows: list[dict], out_html: Path) -> None:
    status_counts = Counter(row["status"] for row in rows)

    def count_table() -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in status_counts.most_common())
        return f"<table><tr><th>Status</th><th>Count</th></tr>{body}</table>"

    body = []
    for row in rows:
        cls = row["status"]
        body.append(
            "<tr>"
            f"<td>{html.escape(row['source_kind'])}</td>"
            f"<td>{html.escape(row['training_labels'])}</td>"
            f"<td>{row['active_mentions']}</td>"
            f"<td>{row['primary_mentions']}</td>"
            f"<td>{row['source_files']}</td>"
            f"<td>{row['dataset_examples']}</td>"
            f"<td class='{cls}'>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(row['next_action'])}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Source Kind Training Coverage</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.missing_training_data, .under_supported, .not_trained_yet {{ background: #ffe7c2; font-weight: bold; }}
.trained_but_not_seen {{ background: #e9f1ff; }}
.ok {{ background: #e9f8ec; }}
</style>
<h1>Source Kind Training Coverage</h1>
<p>This checks whether the source-kind labels shown in the report are backed by actual training folders/examples, and whether they appear in the user's external clips.</p>
{count_table()}
<table>
<tr><th>Source Kind</th><th>Mapped Training Labels</th><th>Active Mentions</th><th>Primary Mentions</th><th>Source Files</th><th>Dataset Examples</th><th>Status</th><th>Next Action</th></tr>
{''.join(body)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=Path("configs/source_kind_training_map.json"))
    parser.add_argument("--source-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--metadata", type=Path, default=Path("data/reference_element_mixture_v5_vocal/metadata.jsonl"))
    parser.add_argument("--source-kind-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_detail.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_training_coverage.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/all_segments_queue/source_kind_training_coverage.html"))
    args = parser.parse_args()

    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    all_training_labels = sorted({label for labels in mapping.values() for label in labels})
    file_counts = count_source_files(args.source_root, all_training_labels)
    example_counts = count_dataset_examples(args.metadata, all_training_labels)

    active_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    if args.source_kind_csv.exists():
        with args.source_kind_csv.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                primary = row.get("primary_source_kind", "")
                if primary:
                    primary_counts[primary] += 1
                active_counts.update(split_pipe(row.get("active_source_kinds", "")))

    rows = []
    for source_kind, training_labels in sorted(mapping.items()):
        source_files = sum(file_counts.get(label, 0) for label in training_labels)
        examples = sum(example_counts.get(label, 0) for label in training_labels)
        active = active_counts[source_kind]
        primary = primary_counts[source_kind]
        status = status_for(source_files, examples, active)
        if status == "missing_training_data":
            next_action = "collect or synthesize examples for this source kind before trusting detections"
        elif status == "under_supported":
            next_action = "add more real examples; current clips mention it often"
        elif status == "not_trained_yet":
            next_action = "run dataset build/train after adding or mapping examples"
        elif status == "trained_but_not_seen":
            next_action = "keep as optional; not a priority for the current clip set"
        else:
            next_action = "covered enough for prototype-level checking"
        rows.append(
            {
                "source_kind": source_kind,
                "training_labels": "|".join(training_labels),
                "active_mentions": active,
                "primary_mentions": primary,
                "source_files": source_files,
                "dataset_examples": examples,
                "status": status,
                "next_action": next_action,
            }
        )

    rows.sort(key=lambda row: (row["status"] == "ok", -int(row["active_mentions"]), row["source_kind"]))
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
