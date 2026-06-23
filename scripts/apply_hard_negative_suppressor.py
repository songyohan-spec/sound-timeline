from __future__ import annotations

import argparse
import csv
import html
import re
from pathlib import Path

import joblib
import numpy as np


PAIR_RE = re.compile(r"([^:;]+):([0-9.]+)/([0-9.]+)")


def parse_candidates(value: str) -> list[dict]:
    items = []
    for part in str(value or "").split(";"):
        part = part.strip()
        match = PAIR_RE.fullmatch(part)
        if not match:
            continue
        items.append(
            {
                "label": match.group(1),
                "score": float(match.group(2)),
                "threshold": float(match.group(3)),
            }
        )
    return items


def format_candidates(items: list[dict]) -> str:
    return "; ".join(f"{item['label']}:{item['score']:.3f}/{item['threshold']:.3f}" for item in items)


def write_html(rows: list[dict], path: Path) -> None:
    body = "".join(
        "<tr>"
        f"<td>{html.escape(row.get('file', ''))}</td>"
        f"<td>{html.escape(str(row.get('start', '')))}-{html.escape(str(row.get('end', '')))}s</td>"
        f"<td>{html.escape(row.get('detected_groups', '') or '-')}</td>"
        f"<td>{html.escape(row.get('detected_labels', '') or '-')}</td>"
        f"<td>{html.escape(row.get('top_labels', ''))}</td>"
        f"<td>{html.escape(row.get('suppressor_distance', ''))}</td>"
        f"<td>{html.escape(row.get('suppressor_applied', '') or '-')}</td>"
        "</tr>"
        for row in rows
    )
    applied = sum(1 for row in rows if row.get("suppressor_applied"))
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Suppressed Ensemble Results</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.note {{ color: #444; }}
</style>
<h1>Suppressed Ensemble Results</h1>
<p>Rows: {len(rows)} | Suppressed rows: {applied}</p>
<p class="note">Hard-negative suppressor lowers specific false-positive labels near reviewed feedback examples.</p>
<table>
<tr><th>File</th><th>Time</th><th>Detected Groups</th><th>Detected Labels</th><th>Top Label Candidates</th><th>Distance</th><th>Suppression</th></tr>
{body}
</table>
</html>"""
    path.write_text(page, encoding="utf-8")


def row_vector(row: dict, suppressor: dict) -> np.ndarray:
    values = []
    aliases = {
        "centroid": "brightness_centroid",
        "flatness": "flatness_noise",
        "motion_strength": "motion_strength",
        "width": "width",
    }
    for name in suppressor["feature_names"]:
        column = aliases.get(name, name)
        try:
            values.append(float(row.get(column, 0.0)))
        except ValueError:
            values.append(0.0)
    return np.array(values, dtype=np.float32)


def nearest_feedback(row: dict, suppressor: dict) -> tuple[dict | None, float]:
    vector = row_vector(row, suppressor)
    z = (vector - suppressor["mean"]) / suppressor["std"]
    best = None
    best_distance = float("inf")
    for example in suppressor["examples"]:
        ex_vec = np.array([float(example["stats"].get(name, 0.0)) for name in suppressor["feature_names"]], dtype=np.float32)
        ex_z = (ex_vec - suppressor["mean"]) / suppressor["std"]
        distance = float(np.sqrt(np.mean((z - ex_z) ** 2)))
        if distance < best_distance:
            best = example
            best_distance = distance
    return best, best_distance


def split_pipe(value: str) -> list[str]:
    return [item.strip() for item in str(value or "").split("|") if item.strip()]


def apply_to_row(row: dict, suppressor: dict) -> dict:
    example, distance = nearest_feedback(row, suppressor)
    row["suppressor_distance"] = "" if example is None else f"{distance:.4f}"
    row["suppressor_applied"] = ""
    if example is None or distance > float(suppressor["radius"]):
        return row

    forbidden_labels = {item.split(":", 1)[1] for item in example["forbidden"] if item.startswith("label:")}
    forbidden_groups = {item.split(":", 1)[1] for item in example["forbidden"] if item.startswith("group:")}
    strength = float(suppressor["strength"])

    candidates = parse_candidates(row.get("top_labels", ""))
    changed = False
    for item in candidates:
        if item["label"] in forbidden_labels:
            item["score"] *= 1.0 - strength
            changed = True
    row["top_labels"] = format_candidates(candidates)

    detected_labels = [label for label in split_pipe(row.get("detected_labels", "")) if label not in forbidden_labels]
    detected_groups = [group for group in split_pipe(row.get("detected_groups", "")) if group not in forbidden_groups]
    if detected_labels != split_pipe(row.get("detected_labels", "")) or detected_groups != split_pipe(row.get("detected_groups", "")):
        changed = True
    row["detected_labels"] = "|".join(detected_labels)
    row["detected_groups"] = "|".join(detected_groups)
    if changed:
        row["suppressor_applied"] = example.get("note", "hard_negative")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/external_reference_ensemble_batch_floor045_strictgroups.csv"))
    parser.add_argument("--suppressor", type=Path, default=Path("models/hard_negative_suppressor.joblib"))
    parser.add_argument("--out", type=Path, default=Path("outputs/external_reference_ensemble_batch_suppressed.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/external_reference_ensemble_batch_suppressed.html"))
    args = parser.parse_args()

    suppressor = joblib.load(args.suppressor)
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [apply_to_row(row, suppressor) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    for extra in ["suppressor_distance", "suppressor_applied"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_html)
    applied = sum(1 for row in rows if row.get("suppressor_applied"))
    print(f"rows: {len(rows)}")
    print(f"suppressed rows: {applied}")
    print(f"wrote: {args.out}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
