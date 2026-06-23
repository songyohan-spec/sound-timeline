from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int = 44_100) -> tuple[np.ndarray, int]:
    import librosa

    try:
        audio, sr = sf.read(path, always_2d=True)
    except Exception:
        loaded, sr = librosa.load(path, sr=None, mono=False)
        if loaded.ndim == 1:
            audio = loaded[:, None]
        else:
            audio = loaded.T

    audio = audio.astype(np.float32)
    if sr != target_sr:
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr))
        min_len = min(len(ch) for ch in channels)
        audio = np.stack([ch[:min_len] for ch in channels], axis=1)
        sr = target_sr
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2], sr


def dsp_cues(audio: np.ndarray, sr: int, start: float, end: float) -> dict:
    import librosa

    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))
    clip = audio[start_sample:end_sample]
    if len(clip) < sr // 8:
        return {
            "dsp_brightness": 0.0,
            "dsp_flatness": 0.0,
            "dsp_stereo_side_ratio": 0.0,
            "dsp_motion_strength": 0.0,
        }

    left = clip[:, 0]
    right = clip[:, 1]
    mono = 0.5 * (left + right)
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)

    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=mono)[0]
    rms = librosa.feature.rms(y=mono, frame_length=2048, hop_length=512)[0]
    if np.max(rms) > 1e-8:
        rms = rms / (np.max(rms) + 1e-8)
    motion_strength = float(np.std(rms))

    mid_rms = float(np.sqrt(np.mean(mid**2) + 1e-8))
    side_rms = float(np.sqrt(np.mean(side**2) + 1e-8))
    return {
        "dsp_brightness": round(float(np.mean(centroid)), 3),
        "dsp_flatness": round(float(np.mean(flatness)), 5),
        "dsp_stereo_side_ratio": round(float(side_rms / (mid_rms + 1e-8)), 4),
        "dsp_motion_strength": round(motion_strength, 4),
    }


def top(segment: dict, key: str) -> tuple[str, float]:
    values = segment["predictions"].get(key, [])
    if not values:
        return "unknown", 0.0
    item = values[0]
    return str(item["label"]), float(item["confidence"])


def second(segment: dict, key: str) -> tuple[str, float]:
    values = segment["predictions"].get(key, [])
    if len(values) < 2:
        return "unknown", 0.0
    item = values[1]
    return str(item["label"]), float(item["confidence"])


def yn(value: bool | None) -> str:
    if value is None:
        return "unclear"
    return "yes" if value else "no"


def dsp_yes_no(value: float, yes_threshold: float, no_threshold: float) -> str:
    if value >= yes_threshold:
        return "yes"
    if value <= no_threshold:
        return "no"
    return "unclear"


def agree(a: str, b: str) -> str:
    if "unclear" in {a, b}:
        return "unclear"
    return "agree" if a == b else "disagree"


def label_segment(segment: dict, confidence_threshold: float, ambiguity_margin: float, cues: dict | None = None) -> dict:
    source, source_conf = top(segment, "source_family")
    source_alt, source_alt_conf = second(segment, "source_family")
    reverb, reverb_conf = top(segment, "reverb")
    distortion, distortion_conf = top(segment, "distortion")
    filter_presence, filter_presence_conf = top(segment, "filter_presence")
    filter_motion, filter_motion_conf = top(segment, "filter_motion_type")
    stereo, stereo_conf = top(segment, "stereo")
    motion, motion_conf = top(segment, "motion_presence")

    source_ambiguous = source_conf < confidence_threshold or (source_conf - source_alt_conf) < ambiguity_margin

    pseudo = {
        "segment_index": segment["index"],
        "start": segment["start"],
        "end": segment["end"],
        "source_primary": source,
        "source_primary_conf": round(source_conf, 4),
        "source_alt": source_alt,
        "source_alt_conf": round(source_alt_conf, 4),
        "source_ambiguous": yn(source_ambiguous),
        "sounds_like_voice": yn(source == "vocal_like" and source_conf >= confidence_threshold),
        "sounds_like_synth": yn(source == "synth" and source_conf >= confidence_threshold),
        "sounds_like_guitar": yn(source == "guitar_like" and source_conf >= confidence_threshold),
        "has_long_tail": yn(reverb in {"long_hall", "washed_out"} and reverb_conf >= confidence_threshold),
        "reverb_label": reverb,
        "reverb_conf": round(reverb_conf, 4),
        "feels_wide": yn(stereo == "wide" and stereo_conf >= confidence_threshold),
        "stereo_label": stereo,
        "stereo_conf": round(stereo_conf, 4),
        "feels_rough_or_crushed": yn(distortion in {"crushed", "heavy_distortion"} and distortion_conf >= confidence_threshold),
        "distortion_label": distortion,
        "distortion_conf": round(distortion_conf, 4),
        "has_filtering": yn(filter_presence == "filtered" and filter_presence_conf >= confidence_threshold),
        "filter_presence_label": filter_presence,
        "filter_presence_conf": round(filter_presence_conf, 4),
        "gets_brighter_or_filter_moves": yn(filter_motion == "dynamic" and filter_motion_conf >= confidence_threshold),
        "filter_motion_label": filter_motion,
        "filter_motion_conf": round(filter_motion_conf, 4),
        "has_pumping_or_motion": yn(motion == "motion" and motion_conf >= confidence_threshold),
        "motion_label": motion,
        "motion_conf": round(motion_conf, 4),
    }

    low_conf = [
        key
        for key, conf in [
            ("source", source_conf),
            ("reverb", reverb_conf),
            ("distortion", distortion_conf),
            ("filter_presence", filter_presence_conf),
            ("filter_motion", filter_motion_conf),
            ("stereo", stereo_conf),
            ("motion", motion_conf),
        ]
        if conf < confidence_threshold
    ]
    pseudo["needs_review"] = yn(bool(low_conf) or source_ambiguous)
    pseudo["low_confidence_fields"] = ";".join(low_conf)
    if cues:
        pseudo.update(cues)
        dsp_wide = dsp_yes_no(float(cues["dsp_stereo_side_ratio"]), yes_threshold=0.18, no_threshold=0.045)
        dsp_motion = dsp_yes_no(float(cues["dsp_motion_strength"]), yes_threshold=0.18, no_threshold=0.06)
        dsp_rough = dsp_yes_no(float(cues["dsp_flatness"]), yes_threshold=0.045, no_threshold=0.012)
        dsp_bright = dsp_yes_no(float(cues["dsp_brightness"]), yes_threshold=2600.0, no_threshold=1200.0)

        pseudo["dsp_feels_wide"] = dsp_wide
        pseudo["dsp_has_motion"] = dsp_motion
        pseudo["dsp_feels_rough_or_noisy"] = dsp_rough
        pseudo["dsp_feels_bright"] = dsp_bright
        pseudo["agreement_wide"] = agree(pseudo["feels_wide"], dsp_wide)
        pseudo["agreement_motion"] = agree(pseudo["has_pumping_or_motion"], dsp_motion)
        pseudo["agreement_rough"] = agree(pseudo["feels_rough_or_crushed"], dsp_rough)

        disagreements = [
            name
            for name in ["wide", "motion", "rough"]
            if pseudo[f"agreement_{name}"] == "disagree"
        ]
        if disagreements:
            pseudo["needs_review"] = "yes"
            current = pseudo["low_confidence_fields"]
            pseudo["low_confidence_fields"] = ";".join([item for item in [current, "dsp_disagreement:" + ",".join(disagreements)] if item])
    return pseudo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Segment inference JSON from infer_audio_segments.py")
    parser.add_argument("--out", type=Path, default=Path("outputs/pseudo_labels.csv"))
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    parser.add_argument("--ambiguity-margin", type=float, default=0.15)
    args = parser.parse_args()

    report = json.loads(args.input.read_text(encoding="utf-8"))
    audio_path = Path(report["audio"])
    audio = None
    sr = None
    if audio_path.exists():
        audio, sr = load_audio(audio_path)
    rows = []
    for segment in report["segments"]:
        cues = dsp_cues(audio, sr, segment["start"], segment["end"]) if audio is not None and sr is not None else None
        rows.append(label_segment(segment, args.confidence_threshold, args.ambiguity_margin, cues))
    if not rows:
        raise SystemExit("No segments found in report.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
