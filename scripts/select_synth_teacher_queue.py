from __future__ import annotations

import argparse
import csv
from pathlib import Path


WATCH_LABELS = {
    "granular_texture",
    "bitcrushed_synth_lead",
    "wavetable_noise",
    "fuzzy_lofi_synth",
    "digital_synth_lead",
    "vocal_synth_hybrid",
    "formant_vocoder",
    "synth_pad_wash",
    "supersaw_stack",
    "synth_pluck_bell",
    "arpeggio_sequence",
}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def key(row: dict) -> tuple[str, str, str, str]:
    return row["track"], row["stem"], str(float(row["start"])), str(float(row["end"]))


def f(row: dict, field: str, default: float = 0.0) -> float:
    try:
        return float(row.get(field, default))
    except (TypeError, ValueError):
        return default


def priority(row: dict) -> tuple[int, float]:
    label = row.get("synth_label_top", "")
    decision = row.get("ensemble_decision", "")
    support = row.get("source_kind_support", "")
    conf = f(row, "synth_label_conf")
    score = 0
    if decision == "needs_review_or_more_data":
        score += 70
    if label in WATCH_LABELS:
        score += 45
    if support in {"unsupported", "family_support"}:
        score += 30
    if row.get("strength") == "strong":
        score += 20
    if decision in {"use_pseudo_label", "use_weak_pseudo_label"}:
        score += 10
    if row.get("stem") in {"vocals", "other", "guitar", "piano"}:
        score += 8
    if label in {"granular_texture", "bitcrushed_synth_lead", "wavetable_noise", "fuzzy_lofi_synth"}:
        score += 12
    return score, conf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synth", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist_v4_cached_batch.csv"))
    parser.add_argument("--ensemble", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v4_cached_aligned_strict_texture.csv"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--per-track", type=int, default=4)
    parser.add_argument("--per-label", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_teacher_queue_v4_strict.csv"))
    args = parser.parse_args()

    ensemble = {key(row): row for row in read_rows(args.ensemble)}
    candidates = []
    for row in read_rows(args.synth):
        joined = dict(row)
        ens = ensemble.get(key(row), {})
        joined["ensemble_decision"] = ens.get("decision", "")
        joined["final_label"] = ens.get("final_label", "")
        joined["source_kind_support"] = ens.get("source_kind_support", "")
        joined["support_matches"] = ens.get("support_matches", "")
        joined["priority_score"], joined["priority_conf"] = priority(joined)
        if joined["priority_score"] > 0 and joined.get("strength") in {"medium", "strong"}:
            candidates.append(joined)

    candidates.sort(key=lambda row: (int(row["priority_score"]), float(row["priority_conf"])), reverse=True)
    selected = []
    by_track: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for row in candidates:
        count = by_track.get(row["track"], 0)
        if count >= args.per_track:
            continue
        label_count = by_label.get(row["synth_label_top"], 0)
        if label_count >= args.per_label:
            continue
        selected.append(row)
        by_track[row["track"]] = count + 1
        by_label[row["synth_label_top"]] = label_count + 1
        if len(selected) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not selected:
        raise SystemExit("No teacher queue rows selected.")
    with args.out.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(selected[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    print(f"selected: {len(selected)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
