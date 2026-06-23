from __future__ import annotations

import argparse
import csv
from pathlib import Path


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def key(file: str, start: str, end: str) -> tuple[str, float, float]:
    return (file, round(float(start), 3), round(float(end), 3))


def read_feedback(path: Path) -> dict[tuple[str, float, float], dict]:
    if not path.exists():
        return {}
    feedback = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            feedback[key(row["file"], row["start"], row["end"])] = row
    return feedback


def filter_pipe(value: str, forbidden: set[str]) -> str:
    return "|".join(item for item in split_pipe(value) if item not in forbidden)


def apply_feedback(row: dict, feedback_row: dict | None) -> dict:
    if not feedback_row:
        row["feedback_applied"] = ""
        return row
    forbidden_groups = set(split_pipe(feedback_row.get("forbid_groups", "")))
    forbidden_labels = set(split_pipe(feedback_row.get("forbid_labels", "")))
    forced_labels = set(split_pipe(feedback_row.get("force_labels", "")))

    row["detected_groups"] = filter_pipe(row.get("detected_groups", ""), forbidden_groups)
    row["detected_labels"] = filter_pipe(row.get("detected_labels", ""), forbidden_labels)
    if forced_labels:
        current = set(split_pipe(row.get("detected_labels", "")))
        row["detected_labels"] = "|".join(sorted(current | forced_labels))
    row["feedback_applied"] = feedback_row.get("note", "yes")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/external_reference_ensemble_batch_floor045_strictgroups.csv"))
    parser.add_argument("--feedback", type=Path, default=Path("configs/segment_feedback.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/external_reference_ensemble_batch_feedback.csv"))
    args = parser.parse_args()

    feedback = read_feedback(args.feedback)
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            fb = feedback.get(key(row["file"], row["start"], row["end"]))
            rows.append(apply_feedback(row, fb))
        fieldnames = list(reader.fieldnames or [])
    if "feedback_applied" not in fieldnames:
        fieldnames.append("feedback_applied")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"feedback rows: {len(feedback)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
