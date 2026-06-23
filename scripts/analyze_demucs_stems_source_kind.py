from __future__ import annotations

import argparse
import csv
import html
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from dsp_palette_score import audio_stats
from infer_reference_elements_timeline import load_audio
from infer_source_kind_model_on_queue import positive_probabilities


DEFAULT_STEM_ORDER = ["vocals", "drums", "bass", "guitar", "piano", "other"]


SYNTH_SOURCE_TOKENS = [
    "synth",
    "wavetable",
    "granular",
    "arpeggio",
    "sequence",
    "vocoder",
    "formant",
    "pad",
    "wash",
    "bell",
    "pluck",
    "lead",
    "aliasing",
]


def discover_stems(stems_root: Path, requested: str) -> list[str]:
    if requested:
        return [stem.strip() for stem in requested.split(",") if stem.strip()]
    found = set()
    for track_dir in sorted(path for path in stems_root.iterdir() if path.is_dir()):
        for wav in track_dir.glob("*.wav"):
            found.add(wav.stem)
    ordered = [stem for stem in DEFAULT_STEM_ORDER if stem in found]
    ordered.extend(sorted(found - set(ordered)))
    return ordered


def source_kind_group(label: str) -> str:
    if "vocal" in label or "rap" in label or "lead_or_hook" in label:
        return "vocal_texture"
    if any(token in label for token in ["drum", "kick", "snare", "hat", "percussion", "breakbeat", "clap"]):
        return "drums"
    if any(token in label for token in ["bass", "808", "sub", "sidechain"]):
        return "bass"
    if "guitar" in label or "string" in label or "violin" in label:
        return "guitar_strings"
    if any(token in label for token in ["piano", "keys", "organ"]):
        return "keys_piano"
    if any(token in label for token in ["sample", "loop", "chopped", "filtered_or_muffled"]):
        return "sample_loop"
    if any(token in label for token in ["noise", "fx", "riser", "impact", "glitch"]):
        return "noise_fx"
    return "synth"


def source_kind_groups(labels: list[str]) -> list[str]:
    return sorted({source_kind_group(label) for label in labels})


def allowed_for_stem(stem: str, label: str) -> bool:
    if stem == "vocals":
        return any(
            token in label
            for token in [
                "vocal",
                "rap",
                "lead_or_hook",
                "chopped_or_stuttered_sample",
            ]
        )
    if stem == "drums":
        return any(
            token in label
            for token in [
                "drum",
                "kick",
                "snare",
                "hat",
                "percussion",
                "breakbeat",
                "clap",
            ]
        )
    if stem == "bass":
        return any(token in label for token in ["bass", "808", "sub", "sidechain"])
    if stem == "guitar":
        return any(
            token in label
            for token in ["guitar", "string", "violin", "sample", "loop", "filtered_or_muffled", *SYNTH_SOURCE_TOKENS]
        )
    if stem == "piano":
        return any(token in label for token in ["piano", "keys", "organ", "sample", "loop", *SYNTH_SOURCE_TOKENS])
    if stem == "other":
        return not any(
            token in label
            for token in [
                "kick",
                "snare",
                "hat",
                "breakbeat",
                "electronic_drum",
                "sub_or_808_bass",
                "distorted_bass",
                "synth_bass",
            ]
        )
    return True


def stem_prior_filter(stem: str, label: str, score: float) -> float:
    if stem == "vocals":
        if any(word in label for word in ["vocal", "rap", "lead_or_hook"]):
            return score * 1.15
        if any(word in label for word in ["drum", "kick", "snare", "hat"]):
            return score * 0.35
    if stem == "drums":
        if any(word in label for word in ["drum", "kick", "snare", "hat", "glitch_percussion"]):
            return score * 1.25
        if any(word in label for word in ["vocal", "synth_pad"]):
            return score * 0.35
    if stem == "bass":
        if any(word in label for word in ["bass", "808", "sub", "sidechain"]):
            return score * 1.25
        if any(word in label for word in ["vocal", "snare", "hat"]):
            return score * 0.35
    if stem == "guitar":
        if any(word in label for word in ["guitar", "string", "violin"]):
            return score * 1.25
        if any(word in label for word in ["drum", "kick", "snare", "hat", "bass", "808"]):
            return score * 0.35
    if stem == "piano":
        if any(word in label for word in ["piano", "keys", "organ", "bell", "pluck"]):
            return score * 1.25
        if any(word in label for word in ["drum", "kick", "snare", "hat", "bass", "808"]):
            return score * 0.35
    if stem == "other":
        if any(word in label for word in ["synth", "guitar", "sample", "keys", "piano", "noise", "fx", "string"]):
            return score * 1.10
    return score


def detect_labels(
    bundle: dict,
    stats: dict[str, float],
    stem: str,
    floor: float,
    top_k: int,
    strict_stem_kind: bool,
    threshold_scale: float,
) -> tuple[list[str], str, int]:
    probs = positive_probabilities(bundle, stats, "labels")
    thresholds = bundle.get("thresholds", {}).get("labels", {})
    adjusted = {label: stem_prior_filter(stem, label, score) for label, score in probs.items()}
    ordered = sorted(adjusted.items(), key=lambda item: item[1], reverse=True)
    detected = []
    suppressed = 0
    for label, score in ordered:
        if strict_stem_kind and not allowed_for_stem(stem, label):
            suppressed += 1
            continue
        threshold = max(float(thresholds.get(label, floor)) * threshold_scale, floor)
        if score >= threshold:
            detected.append(label)
    if strict_stem_kind and not detected:
        for label, score in ordered:
            if allowed_for_stem(stem, label):
                threshold = max(float(thresholds.get(label, floor)) * threshold_scale, floor) * 0.8
                if score >= threshold:
                    detected.append(label)
                    break
    top = "; ".join(
        f"{label}:{score:.3f}/{max(float(thresholds.get(label, floor)) * threshold_scale, floor):.3f}{'/detected' if label in detected else '/possible'}"
        for label, score in ordered
        if not strict_stem_kind or allowed_for_stem(stem, label)
    )
    return detected[:top_k], "; ".join(top.split("; ")[:top_k]), suppressed


def positive_probabilities_batch(bundle: dict, stat_rows: list[dict[str, float]], output_name: str) -> list[dict[str, float]]:
    entry = bundle["outputs"][output_name]
    model = entry["model"]
    binarizer = entry["binarizer"]
    cols = bundle["feature_cols"]
    x = np.array([[features.get(col, 0.0) for col in cols] for features in stat_rows], dtype=np.float32)
    x = bundle["scaler"].transform(x)
    raw = model.predict_proba(x)
    by_label = {}
    for idx, class_probs in enumerate(raw):
        classes = model.classes_[idx]
        if len(classes) == 1:
            prob = np.ones(len(stat_rows), dtype=np.float32) * float(classes[0])
        else:
            one_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            prob = class_probs[:, one_index].astype(np.float32)
        by_label[str(binarizer.classes_[idx])] = prob
    out = []
    for row_idx in range(len(stat_rows)):
        out.append({label: float(values[row_idx]) for label, values in by_label.items()})
    return out


def detect_labels_from_probs(
    bundle: dict,
    probs: dict[str, float],
    stem: str,
    floor: float,
    top_k: int,
    strict_stem_kind: bool,
    threshold_scale: float,
) -> tuple[list[str], str, int]:
    thresholds = bundle.get("thresholds", {}).get("labels", {})
    adjusted = {label: stem_prior_filter(stem, label, score) for label, score in probs.items()}
    ordered = sorted(adjusted.items(), key=lambda item: item[1], reverse=True)
    detected = []
    suppressed = 0
    for label, score in ordered:
        if strict_stem_kind and not allowed_for_stem(stem, label):
            suppressed += 1
            continue
        threshold = max(float(thresholds.get(label, floor)) * threshold_scale, floor)
        if score >= threshold:
            detected.append(label)
    if strict_stem_kind and not detected:
        for label, score in ordered:
            if allowed_for_stem(stem, label):
                threshold = max(float(thresholds.get(label, floor)) * threshold_scale, floor) * 0.8
                if score >= threshold:
                    detected.append(label)
                    break
    top = "; ".join(
        f"{label}:{score:.3f}/{max(float(thresholds.get(label, floor)) * threshold_scale, floor):.3f}{'/detected' if label in detected else '/possible'}"
        for label, score in ordered
        if not strict_stem_kind or allowed_for_stem(stem, label)
    )
    return detected[:top_k], "; ".join(top.split("; ")[:top_k]), suppressed


def segment_stem(
    stem_path: Path,
    segment_seconds: float,
    hop_seconds: float,
    model_bundle: dict,
    floor: float,
    top_k: int,
    quality: str,
    strict_stem_kind: bool,
    threshold_scale: float,
) -> list[dict]:
    audio, sr = load_audio(stem_path)
    segment_len = max(1, int(segment_seconds * sr))
    hop_len = max(1, int(hop_seconds * sr))
    duration = len(audio) / sr
    starts = list(range(0, max(1, len(audio) - segment_len + 1), hop_len)) or [0]
    rows = []
    stem = stem_path.stem
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for index, start in enumerate(starts):
            end = min(len(audio), start + segment_len)
            clip = audio[start:end]
            if len(clip) < segment_len:
                pad = np.zeros((segment_len - len(clip), clip.shape[1]), dtype=np.float32)
                clip = np.vstack([clip, pad])
            clip_path = tmp_dir / f"{stem}_{index:04d}.wav"
            sf.write(clip_path, clip, sr)
            stats = audio_stats(clip_path, quality=quality)
            detected, top, suppressed = detect_labels(model_bundle, stats, stem, floor, top_k, strict_stem_kind, threshold_scale)
            rows.append(
                {
                    "stem": stem,
                    "start": round(start / sr, 4),
                    "end": round(min(duration, (start + segment_len) / sr), 4),
                    "detected_source_kinds": "|".join(detected),
                    "detected_source_groups": "|".join(source_kind_groups(detected)),
                    "top_source_kinds": top,
                    "suppressed_cross_stem_labels": suppressed,
                    "centroid": round(stats["centroid"], 4),
                    "flatness": round(stats["flatness"], 6),
                    "motion_strength": round(stats["motion_strength"], 6),
                    "width": round(stats["width"], 6),
                }
            )
    return rows


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


def rows_from_feature_cache(
    cache_path: Path,
    model_bundle: dict,
    floor: float,
    top_k: int,
    strict_stem_kind: bool,
    threshold_scale: float,
) -> list[dict]:
    rows = []
    with cache_path.open("r", encoding="utf-8-sig", newline="") as file:
        cached_rows = list(csv.DictReader(file))
    stat_rows = [stats_from_cache_row(row) for row in cached_rows]
    prob_rows = positive_probabilities_batch(model_bundle, stat_rows, "labels")
    for cached, stats, probs in zip(cached_rows, stat_rows, prob_rows):
            detected, top, suppressed = detect_labels_from_probs(model_bundle, probs, cached["stem"], floor, top_k, strict_stem_kind, threshold_scale)
            rows.append(
                {
                    "stem": cached["stem"],
                    "start": cached["start"],
                    "end": cached["end"],
                    "detected_source_kinds": "|".join(detected),
                    "detected_source_groups": "|".join(source_kind_groups(detected)),
                    "top_source_kinds": top,
                    "suppressed_cross_stem_labels": suppressed,
                    "centroid": round(stats.get("centroid", 0.0), 4),
                    "flatness": round(stats.get("flatness", 0.0), 6),
                    "motion_strength": round(stats.get("motion_strength", 0.0), 6),
                    "width": round(stats.get("width", 0.0), 6),
                    "track": cached["track"],
                    "stem_path": cached["stem_path"],
                }
            )
    return rows


def write_html(rows: list[dict], out_html: Path, title: str, stem_root: Path) -> None:
    stem_counts = Counter()
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track"]].append(row)
        stem_counts.update(row["detected_source_kinds"].split("|") if row["detected_source_kinds"] else [])

    count_body = "".join(f"<tr><td>{html.escape(label)}</td><td>{count}</td></tr>" for label, count in stem_counts.most_common())
    sections = []
    for track, track_rows in sorted(by_track.items()):
        detail_rows = []
        for row in track_rows:
            audio_rel = Path(row["stem_path"]).relative_to(out_html.parent).as_posix() if Path(row["stem_path"]).is_relative_to(out_html.parent) else Path(row["stem_path"]).as_posix()
            detail_rows.append(
                "<tr>"
                f"<td>{html.escape(row['stem'])}</td>"
                f"<td>{row['start']}-{row['end']}s</td>"
                f"<td><audio controls preload='metadata' src='{html.escape(audio_rel)}'></audio></td>"
                f"<td>{html.escape(row.get('detected_source_groups', '') or '-')}</td>"
                f"<td>{html.escape(row['detected_source_kinds'] or '-')}</td>"
                f"<td>{html.escape(row['top_source_kinds'])}</td>"
                f"<td>{row['suppressed_cross_stem_labels']}</td>"
                f"<td>{row['centroid']}</td>"
                f"<td>{row['flatness']}</td>"
                f"<td>{row['motion_strength']}</td>"
                "</tr>"
            )
        sections.append(
            f"""<section>
<h2>{html.escape(track)}</h2>
<table>
<tr><th>Stem</th><th>Time</th><th>Stem Audio</th><th>Source Groups</th><th>Detected Source Kinds</th><th>Top Source Kinds</th><th>Suppressed</th><th>Brightness</th><th>Noise</th><th>Motion</th></tr>
{''.join(detail_rows)}
</table>
</section>"""
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 180px; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
</style>
<h1>{html.escape(title)}</h1>
<p>Demucs separates each clip into stems. Source-kind predictions are then run per stem, so dense mixed clips are no longer treated as one blob.</p>
<h2>Detected Source Kind Counts</h2>
<table><tr><th>Source Kind</th><th>Count</th></tr>{count_body}</table>
{''.join(sections)}
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_test/htdemucs"))
    parser.add_argument("--model", type=Path, default=Path("models/source_kind_multilabel_v1.joblib"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--floor", type=float, default=0.30)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--strict-stem-kind", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--stems", default="", help="Comma-separated stem names. Defaults to all wav stems found under --stems-root.")
    parser.add_argument("--feature-cache", type=Path, default=None, help="Optional cached stem segment feature CSV from export_demucs_stem_feature_cache.py.")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_test/stem_source_kind.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_test/stem_source_kind.html"))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    if args.feature_cache:
        rows = rows_from_feature_cache(args.feature_cache, bundle, args.floor, args.top_k, args.strict_stem_kind, args.threshold_scale)
    else:
        rows = []
        stems = discover_stems(args.stems_root, args.stems)
        for track_dir in sorted(path for path in args.stems_root.iterdir() if path.is_dir()):
            for stem in stems:
                stem_path = track_dir / f"{stem}.wav"
                if not stem_path.exists():
                    continue
                for row in segment_stem(
                    stem_path,
                    args.segment_seconds,
                    args.hop_seconds,
                    bundle,
                    args.floor,
                    args.top_k,
                    args.quality,
                    args.strict_stem_kind,
                    args.threshold_scale,
                ):
                    row["track"] = track_dir.name
                    row["stem_path"] = str(stem_path)
                    rows.append(row)
    if not rows:
        raise SystemExit(f"No stem wav files found under {args.stems_root}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_html(rows, args.out_html, "Demucs Stem Source-Kind Analysis", args.stems_root)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
