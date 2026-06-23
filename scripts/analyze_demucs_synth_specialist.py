from __future__ import annotations

import argparse
import csv
import html
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_demucs_stems_source_kind import DEFAULT_STEM_ORDER, discover_stems
from dsp_palette_score import audio_stats
from infer_reference_elements_timeline import load_audio


def top_probs(model: dict, x: np.ndarray, output: str, top_k: int) -> list[tuple[str, float]]:
    item = model["outputs"][output]
    estimator = item["model"]
    encoder = item["encoder"]
    probs = estimator.predict_proba(x)[0]
    order = np.argsort(probs)[::-1][:top_k]
    return [(str(encoder.classes_[idx]), float(probs[idx])) for idx in order]


def top_probs_batch(model: dict, x: np.ndarray, output: str, top_k: int) -> list[list[tuple[str, float]]]:
    item = model["outputs"][output]
    estimator = item["model"]
    encoder = item["encoder"]
    probs = estimator.predict_proba(x)
    out = []
    for row in probs:
        order = np.argsort(row)[::-1][:top_k]
        out.append([(str(encoder.classes_[idx]), float(row[idx])) for idx in order])
    return out


def write_segment(path: Path, audio: np.ndarray, sr: int, start: float, end: float) -> Path:
    start_i = max(0, int(start * sr))
    end_i = min(audio.shape[0], int(end * sr))
    segment = audio[start_i:end_i]
    if segment.shape[0] < int(0.25 * sr):
        raise ValueError("segment too short")
    sf.write(path, segment, sr)
    return path


def analyze_segment(model: dict, feature_cols: list[str], segment_path: Path, quality: str) -> dict:
    stats = audio_stats(segment_path, quality=quality)
    return analyze_stats(model, feature_cols, stats)


def analyze_stats(model: dict, feature_cols: list[str], stats: dict[str, float]) -> dict:
    x = np.array([[float(stats.get(col, 0.0)) for col in feature_cols]], dtype=np.float32)
    x = model["scaler"].transform(x)
    family = top_probs(model, x, "family", 3)
    label = top_probs(model, x, "label", 5)
    return {
        "synth_family_top": family[0][0],
        "synth_family_conf": round(family[0][1], 6),
        "synth_family_alternatives": "|".join(f"{name}:{score:.3f}" for name, score in family[1:]),
        "synth_label_top": label[0][0],
        "synth_label_conf": round(label[0][1], 6),
        "synth_label_alternatives": "|".join(f"{name}:{score:.3f}" for name, score in label[1:]),
    }


def analyze_stats_batch(model: dict, feature_cols: list[str], stat_rows: list[dict[str, float]]) -> list[dict]:
    x = np.array([[float(stats.get(col, 0.0)) for col in feature_cols] for stats in stat_rows], dtype=np.float32)
    x = model["scaler"].transform(x)
    families = top_probs_batch(model, x, "family", 3)
    labels = top_probs_batch(model, x, "label", 5)
    out = []
    for family, label in zip(families, labels):
        out.append(
            {
                "synth_family_top": family[0][0],
                "synth_family_conf": round(family[0][1], 6),
                "synth_family_alternatives": "|".join(f"{name}:{score:.3f}" for name, score in family[1:]),
                "synth_label_top": label[0][0],
                "synth_label_conf": round(label[0][1], 6),
                "synth_label_alternatives": "|".join(f"{name}:{score:.3f}" for name, score in label[1:]),
            }
        )
    return out


def stats_from_cache_row(row: dict) -> dict[str, float]:
    meta = {"track", "stem", "start", "end", "stem_path", "duration"}
    out = {}
    for key, value in row.items():
        if key in meta:
            continue
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


NON_BASS_SYNTH_LABELS = {
    "synth_pad_wash",
    "supersaw_stack",
    "digital_synth_lead",
    "bitcrushed_synth_lead",
    "synth_pluck_bell",
    "arpeggio_sequence",
    "granular_texture",
    "wavetable_noise",
    "fuzzy_lofi_synth",
    "synth_flute_pipe",
    "vocal_synth_hybrid",
    "formant_vocoder",
}

BASS_SYNTH_LABELS = {"synth_bass", "sidechained_synth_bass", "sub_808_synth_bass"}


def reconcile(row: dict, min_conf: float) -> tuple[str, str]:
    family = row["synth_family_top"]
    label = row["synth_label_top"]
    stem = row["stem"]
    family_conf = float(row["synth_family_conf"])
    label_conf = float(row["synth_label_conf"])

    if family == "not_synth" or label.startswith("not_synth"):
        return "not_synth", "family_or_label_not_synth"

    if stem == "drums" and (family_conf < 0.92 or label_conf < 0.58):
        return "not_synth", "drum_stem_gate"

    if stem == "vocals" and label not in {"vocal_synth_hybrid", "formant_vocoder", "synth_pluck_bell", "digital_synth_lead"}:
        if label_conf < 0.58:
            return "weak", "vocal_stem_non_vocal_synth_gate"

    if stem == "bass" and label not in BASS_SYNTH_LABELS:
        if label_conf < 0.50:
            return "weak", "bass_stem_non_bass_synth_gate"

    if stem in {"guitar", "piano", "other"} and label in BASS_SYNTH_LABELS:
        if label_conf < 0.56:
            return "weak", "non_bass_stem_bass_synth_gate"

    if family_conf < 0.58 or label_conf < min_conf:
        return "weak", "low_confidence"
    if label_conf >= 0.58 and family_conf >= 0.76:
        return "strong", "high_confidence"
    return "medium", "usable_confidence"


def write_html(rows: list[dict], out_html: Path) -> None:
    body = []
    for row in rows:
        cls = row["strength"]
        body.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{html.escape(row['stem'])}</td>"
            f"<td>{row['start']}-{row['end']}s</td>"
            f"<td class='{cls}'>{html.escape(row['strength'])}</td>"
            f"<td>{html.escape(row['synth_family_top'])} ({row['synth_family_conf']})</td>"
            f"<td>{html.escape(row['synth_label_top'])} ({row['synth_label_conf']})</td>"
            f"<td>{html.escape(row['synth_label_alternatives'])}</td>"
            f"<td>{html.escape(row.get('gate_reason', ''))}</td>"
            f"<td><audio controls preload='none' src='{html.escape(row['stem_path'])}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Synth Specialist</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
.strong {{ background: #e8f7ed; }}
.medium {{ background: #fff7df; }}
.weak {{ background: #ffe9dc; }}
.not_synth {{ background: #f4f4f4; color: #777; }}
.note {{ color: #444; max-width: 980px; }}
</style>
<h1>Demucs Synth Specialist</h1>
<p class="note">A focused synthetic/hard-negative model for synth-like material. It scans every Demucs stem segment, so synths can appear under bass, piano, guitar, other, or vocals.</p>
<table>
<tr><th>Track</th><th>Stem</th><th>Time</th><th>Strength</th><th>Family</th><th>Label</th><th>Alternatives</th><th>Gate</th><th>Stem Audio</th></tr>
{''.join(body)}
</table>
<p class="note">Caution: this is a synth-specialist hypothesis layer. It improves recall for synth-like sounds but still cannot recover exact VST/preset/oscillator settings.</p>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_6s_full/htdemucs_6s"))
    parser.add_argument("--model", type=Path, default=Path("models/synth_specialist_v1.joblib"))
    parser.add_argument("--stems", default="")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--min-conf", type=float, default=0.34)
    parser.add_argument("--feature-cache", type=Path, default=None, help="Optional cached stem segment feature CSV from export_demucs_stem_feature_cache.py.")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_specialist.html"))
    args = parser.parse_args()

    model = joblib.load(args.model)
    feature_cols = model["feature_cols"]
    rows = []
    if args.feature_cache:
        with args.feature_cache.open("r", encoding="utf-8-sig", newline="") as file:
            cached_rows = list(csv.DictReader(file))
        results = analyze_stats_batch(model, feature_cols, [stats_from_cache_row(row) for row in cached_rows])
        for cached, result in zip(cached_rows, results):
                row = {
                    "track": cached["track"],
                    "stem": cached["stem"],
                    "start": cached["start"],
                    "end": cached["end"],
                    "stem_path": cached["stem_path"],
                    **result,
                }
                row["strength"], row["gate_reason"] = reconcile(row, args.min_conf)
                rows.append(row)
    else:
        stems = discover_stems(args.stems_root, args.stems)
        stem_order = [stem for stem in DEFAULT_STEM_ORDER if stem in stems] + sorted(set(stems) - set(DEFAULT_STEM_ORDER))
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for track_dir in sorted(path for path in args.stems_root.iterdir() if path.is_dir()):
                for stem in stem_order:
                    stem_path = track_dir / f"{stem}.wav"
                    if not stem_path.exists():
                        continue
                    audio, sr = load_audio(stem_path, sample_rate=22050)
                    duration = audio.shape[0] / sr
                    start = 0.0
                    while start < duration - 0.25:
                        end = min(duration, start + args.segment_seconds)
                        segment_path = tmp_dir / f"{track_dir.name}_{stem}_{start:.2f}.wav"
                        try:
                            write_segment(segment_path, audio, sr, start, end)
                            result = analyze_segment(model, feature_cols, segment_path, args.quality)
                        except Exception as exc:
                            result = {
                                "synth_family_top": "error",
                                "synth_family_conf": 0.0,
                                "synth_family_alternatives": "",
                                "synth_label_top": f"error:{type(exc).__name__}",
                                "synth_label_conf": 0.0,
                                "synth_label_alternatives": "",
                            }
                        row = {
                            "track": track_dir.name,
                            "stem": stem,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "stem_path": stem_path.as_posix(),
                            **result,
                        }
                        row["strength"], row["gate_reason"] = reconcile(row, args.min_conf)
                        rows.append(row)
                        start += args.hop_seconds

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
