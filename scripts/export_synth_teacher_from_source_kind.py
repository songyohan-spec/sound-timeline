from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import soundfile as sf

from infer_reference_elements_timeline import load_audio


SOURCE_TO_SYNTH = {
    "synth_pad_or_wash": "synth_pad_wash",
    "supersaw_or_bright_synth_stack": "supersaw_stack",
    "digital_synth_lead": "digital_synth_lead",
    "bitcrushed_or_aliasing_synth": "bitcrushed_synth_lead",
    "arpeggio_or_sequence_synth": "arpeggio_sequence",
    "synth_pluck_or_bell": "synth_pluck_bell",
    "granular_or_resampled_synth": "granular_texture",
    "wavetable_noise_synth": "wavetable_noise",
    "fuzzy_distorted_synth": "fuzzy_lofi_synth",
    "vocal_synth_hybrid": "vocal_synth_hybrid",
    "formant_or_vocoder_vocal": "formant_vocoder",
    "synth_bass": "synth_bass",
    "sidechained_bass_pulse": "sidechained_synth_bass",
    "sub_or_808_bass": "sub_808_synth_bass",
}

BASS_LABELS = {"synth_bass", "sidechained_synth_bass", "sub_808_synth_bass"}


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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
            out[label.strip()] = (float(bits[0]), float(bits[1]), bits[2].strip())
        except ValueError:
            continue
    return out


def family_for(label: str) -> str:
    return "synth_bass" if label in BASS_LABELS else "synth"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value).strip("_")


def export_clip(row: dict, out_file: Path, sample_rate: int) -> float:
    stem_path = Path(row["stem_path"])
    audio, sr = load_audio(stem_path, sample_rate=sample_rate)
    start = float(row["start"])
    end = float(row["end"])
    start_i = max(0, int(start * sr))
    end_i = min(audio.shape[0], int(end * sr))
    clip = audio[start_i:end_i]
    if clip.shape[0] < int(0.25 * sr):
        raise ValueError("segment too short")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_file, clip, sr)
    return clip.shape[0] / sr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-kind", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind_merged_v3.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/synth_source_kind_teacher_v1"))
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--min-ratio", type=float, default=1.03)
    parser.add_argument("--max-per-label", type=int, default=80)
    args = parser.parse_args()

    candidates = []
    for row in read_rows(args.source_kind):
        top = parse_top(row.get("top_source_kinds", ""))
        detected = set(str(row.get("detected_source_kinds", "")).split("|"))
        for source_kind, synth_label in SOURCE_TO_SYNTH.items():
            if source_kind not in top:
                continue
            score, threshold, status = top[source_kind]
            ratio = score / max(threshold, 1e-6)
            if source_kind not in detected and status != "detected" and ratio < args.min_ratio:
                continue
            candidates.append((ratio, score, source_kind, synth_label, row))

    candidates.sort(key=lambda item: (item[3], -item[0], -item[1]))
    counts = defaultdict(int)
    metadata = []
    audio_dir = args.out / "audio"
    skipped = 0
    for ratio, score, source_kind, synth_label, row in candidates:
        if counts[synth_label] >= args.max_per_label:
            continue
        idx = counts[synth_label]
        out_file = audio_dir / f"teacher_{safe_name(synth_label)}_{idx:03d}_{safe_name(row['track'])}_{safe_name(row['stem'])}_{float(row['start']):.2f}.wav"
        try:
            duration = export_clip(row, out_file, args.sample_rate)
        except Exception:
            skipped += 1
            continue
        counts[synth_label] += 1
        metadata.append(
            {
                "file": out_file.relative_to(args.out).as_posix(),
                "label": synth_label,
                "family": family_for(synth_label),
                "base_label": f"source_kind_teacher:{source_kind}",
                "duration": round(duration, 4),
                "source_index": len(metadata),
                "source_kind": source_kind,
                "source_kind_score": round(score, 6),
                "source_kind_ratio": round(ratio, 6),
                "track": row["track"],
                "stem": row["stem"],
                "start": row["start"],
                "end": row["end"],
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "metadata.jsonl").open("w", encoding="utf-8") as file:
        for row in metadata:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("exported:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")
    print(f"skipped: {skipped}")
    print(f"metadata: {args.out / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
