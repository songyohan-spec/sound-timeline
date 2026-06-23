from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf

from infer_reference_elements_timeline import load_audio


FAMILY_BY_LABEL = {
    "synth_bass": "synth_bass",
    "sidechained_synth_bass": "synth_bass",
    "sub_808_synth_bass": "synth_bass",
}


def read_csv(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def family_for(label: str) -> str:
    return FAMILY_BY_LABEL.get(label, "synth")


def write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_clip(row: dict, out_file: Path, sample_rate: int) -> float:
    stem_path = Path(row["stem_path"]) if "stem_path" in row else None
    if stem_path is None or not stem_path.exists():
        stem_path = Path("outputs/demucs_stems_6s_full/htdemucs_6s") / row["track"] / f"{row['stem']}.wav"
    audio, sr = load_audio(stem_path, sample_rate=sample_rate)
    start = float(row["start"])
    end = float(row["end"])
    start_i = max(0, int(start * sr))
    end_i = min(audio.shape[0], int(end * sr))
    clip = audio[start_i:end_i]
    if clip.shape[0] < int(0.25 * sr):
        raise ValueError(f"segment too short: {row['track']} {row['stem']} {start}-{end}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_file, clip, sr)
    return clip.shape[0] / sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_fast_ensemble.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/synth_pseudo_real_v1"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--decisions", default="use_pseudo_label,use_weak_pseudo_label")
    args = parser.parse_args()

    allowed = {item.strip() for item in args.decisions.split(",") if item.strip()}
    source_rows = [row for row in read_csv(args.ensemble) if row.get("decision") in allowed and row.get("final_label") != "ambiguous"]
    if not source_rows:
        raise SystemExit("No pseudo-label rows selected.")

    audio_dir = args.out / "audio"
    rows = []
    skipped = 0
    for idx, row in enumerate(source_rows):
        label = row["final_label"]
        out_file = audio_dir / f"pseudo_{idx:06d}_{label}_{row['track']}_{row['stem']}_{float(row['start']):.2f}.wav"
        try:
            duration = export_clip(row, out_file, args.sample_rate)
        except Exception as exc:
            print(f"skip {idx}: {exc}")
            skipped += 1
            continue
        rows.append(
            {
                "file": out_file.relative_to(args.out).as_posix(),
                "label": label,
                "family": family_for(label),
                "base_label": f"pseudo_real:{row['track']}:{row['stem']}",
                "duration": round(duration, 4),
                "source_index": idx,
                "pseudo_confidence": row.get("specialist_conf", ""),
                "support": row.get("source_kind_support", ""),
                "support_matches": row.get("support_matches", ""),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(rows, args.out / "metadata.jsonl")
    print(f"selected: {len(source_rows)}")
    print(f"exported: {len(rows)}")
    print(f"skipped: {skipped}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
