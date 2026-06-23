from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def row_key(row: dict) -> tuple[str, str, str, str]:
    return row["track"], row["stem"], str(float(row["start"])), str(float(row["end"]))


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def parse_top(value: str) -> dict[str, tuple[float, float, str]]:
    out = {}
    for part in str(value or "").split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        label, rest = part.split(":", 1)
        bits = rest.split("/")
        if len(bits) < 3:
            continue
        try:
            score = float(bits[0])
            threshold = float(bits[1])
        except ValueError:
            continue
        out[label.strip()] = (score, threshold, bits[2].strip())
    return out


def format_top(scores: dict[str, tuple[float, float, str]], limit: int) -> str:
    ranked = sorted(scores.items(), key=lambda item: item[1][0], reverse=True)
    return "; ".join(f"{label}:{score:.3f}/{threshold:.3f}/{status}" for label, (score, threshold, status) in ranked[:limit])


def merge_rows(rows: list[dict], top_k: int) -> dict:
    base = dict(rows[0])
    detected = set()
    groups = set()
    top_scores: dict[str, tuple[float, float, str]] = {}
    suppressed = 0
    numeric_fields = ["centroid", "flatness", "motion_strength", "width", "bandwidth", "rolloff", "zcr"]

    for row in rows:
        detected.update(split_pipe(row.get("detected_source_kinds", "")))
        groups.update(split_pipe(row.get("detected_source_groups", "")))
        suppressed += int(float(row.get("suppressed_cross_stem_labels", 0) or 0))
        for label, candidate in parse_top(row.get("top_source_kinds", "")).items():
            if label not in top_scores or candidate[0] > top_scores[label][0]:
                top_scores[label] = candidate

    base["detected_source_kinds"] = "|".join(sorted(detected))
    base["detected_source_groups"] = "|".join(sorted(groups))
    base["top_source_kinds"] = format_top(top_scores, top_k)
    base["suppressed_cross_stem_labels"] = suppressed
    base["evidence_models_merged"] = len(rows)
    for field in numeric_fields:
        values = []
        for row in rows:
            try:
                values.append(float(row.get(field, "")))
            except ValueError:
                pass
        if values:
            base[field] = round(sum(values) / len(values), 6)
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=14)
    args = parser.parse_args()

    grouped: dict[tuple[str, str, str, str], list[dict]] = {}
    for path in args.inputs:
        for row in read_rows(path):
            grouped.setdefault(row_key(row), []).append(row)

    rows = [merge_rows(items, args.top_k) for _, items in sorted(grouped.items())]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows: {len(rows)}")
    print(f"wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
