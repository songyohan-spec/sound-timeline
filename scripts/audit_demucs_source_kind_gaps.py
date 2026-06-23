from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}
DEFAULT_STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


def split_pipe(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def stem_order(counts: dict[str, Counter[str]]) -> list[str]:
    found = set(counts)
    ordered = [stem for stem in DEFAULT_STEM_ORDER if stem in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def count_source_files(root: Path, labels: list[str]) -> dict[str, int]:
    out = {}
    for label in labels:
        folder = root / label
        out[label] = len([path for path in folder.rglob("*") if path.suffix.lower() in AUDIO_EXTENSIONS]) if folder.exists() else 0
    return out


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
            for label in row_labels:
                if label in counts:
                    counts[label] += 1
    return counts


def status_for(mentions: int, source_files: int, examples: int, empty_stem_rows: int) -> tuple[str, str]:
    if mentions == 0:
        return "not_seen", "not currently visible in the Demucs stem set"
    if source_files == 0 and examples == 0:
        return "missing_training_data", "add real/open-source examples or map this label to an existing trained target"
    if source_files < 3 and mentions >= 10:
        return "under_supported_sources", "add diverse real source files; current mapping is too bootstrap-heavy"
    if examples < 50 and mentions >= 10:
        return "under_trained", "generate/rebuild more mixture examples for this source kind"
    if empty_stem_rows >= 20 and mentions < 5:
        return "weak_detector_for_stem", "stem has many empty rows; lower thresholds or add stem-specific training"
    return "ok", "prototype coverage is acceptable"


def write_html(rows: list[dict], empty_counts: Counter[str], out_html: Path) -> None:
    status_counts = Counter(row["status"] for row in rows)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Label</th><th>Count</th></tr>{body}</table></section>"

    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(row['source_kind'])}</td>"
            f"<td>{html.escape(row['stem_mentions'])}</td>"
            f"<td>{row['total_mentions']}</td>"
            f"<td>{html.escape(row['training_labels'])}</td>"
            f"<td>{row['source_files']}</td>"
            f"<td>{row['dataset_examples']}</td>"
            f"<td class='{html.escape(row['status'])}'>{html.escape(row['status'])}</td>"
            f"<td>{html.escape(row['next_action'])}</td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Source-Kind Gap Audit</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.missing_training_data, .under_supported_sources, .under_trained, .weak_detector_for_stem {{ background: #ffe7c2; font-weight: bold; }}
.not_seen {{ background: #f2f2f2; color: #555; }}
.ok {{ background: #e9f8ec; }}
</style>
<h1>Demucs Source-Kind Gap Audit</h1>
<p>This audit uses the strict Demucs stem analysis, not the older mixed-clip source-kind report.</p>
{count_table("Status", status_counts)}
{count_table("Empty Stem Rows", empty_counts)}
<table>
<tr><th>Source Kind</th><th>Stem Mentions</th><th>Total Mentions</th><th>Mapped Training Labels</th><th>Source Files</th><th>Dataset Examples</th><th>Status</th><th>Next Action</th></tr>
{''.join(body)}
</table>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=Path, default=Path("configs/source_kind_training_map.json"))
    parser.add_argument("--stem-csv", type=Path, default=Path("outputs/demucs_stems_full/stem_source_kind_strict.csv"))
    parser.add_argument("--source-root", type=Path, default=Path("data/reference_training_sources"))
    parser.add_argument("--metadata", type=Path, default=Path("data/reference_element_mixture_v5_vocal/metadata.jsonl"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_full/source_kind_gap_audit.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_full/source_kind_gap_audit.html"))
    args = parser.parse_args()

    mapping = json.loads(args.map.read_text(encoding="utf-8"))
    all_training_labels = sorted({label for labels in mapping.values() for label in labels})
    source_file_counts = count_source_files(args.source_root, all_training_labels)
    example_counts = count_dataset_examples(args.metadata, all_training_labels)

    stem_counts: dict[str, Counter[str]] = defaultdict(Counter)
    empty_counts: Counter[str] = Counter()
    with args.stem_csv.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            stem = row["stem"]
            labels = split_pipe(row.get("detected_source_kinds", ""))
            if not labels:
                empty_counts[stem] += 1
            for label in labels:
                stem_counts[stem][label] += 1

    rows = []
    stems = stem_order(stem_counts)
    for source_kind, training_labels in mapping.items():
        total = sum(stem_counts[stem][source_kind] for stem in stems)
        stem_mentions = ", ".join(f"{stem}:{stem_counts[stem][source_kind]}" for stem in stems if stem_counts[stem][source_kind])
        source_files = sum(source_file_counts.get(label, 0) for label in training_labels)
        examples = sum(example_counts.get(label, 0) for label in training_labels)
        status, next_action = status_for(total, source_files, examples, max(empty_counts.values() or [0]))
        rows.append(
            {
                "source_kind": source_kind,
                "stem_mentions": stem_mentions or "-",
                "total_mentions": total,
                "training_labels": "|".join(training_labels),
                "source_files": source_files,
                "dataset_examples": examples,
                "status": status,
                "next_action": next_action,
            }
        )

    priority = {
        "missing_training_data": 0,
        "under_supported_sources": 1,
        "under_trained": 2,
        "weak_detector_for_stem": 3,
        "ok": 4,
        "not_seen": 5,
    }
    rows.sort(key=lambda row: (priority.get(row["status"], 9), -int(row["total_mentions"]), row["source_kind"]))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, empty_counts, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
