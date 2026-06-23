from __future__ import annotations

import argparse
import json
from pathlib import Path


MIN_SOURCE_SCORE = 0.015
MIN_FX_TEXTURE_SCORE = 0.05
MIN_PROCESSING_SCORE = 0.01
SOURCE_FAMILIES = {"vocal_derived", "synth_derived", "guitar_derived", "hybrid_sampled"}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def top_palette(palette: dict, family: str, n: int = 3) -> list[dict]:
    return palette.get("families", {}).get(family, [])[:n]


def top_overall(palette: dict, n: int = 10) -> list[dict]:
    return palette.get("top_overall", [])[:n]


def top_across_families(palette: dict, families: set[str], n: int = 10) -> list[dict]:
    items = [
        item
        for item in palette.get("top_overall", [])
        if item.get("family") in families
    ]
    return items[:n]


def rf_top(segments: dict, key: str) -> dict | None:
    rows = segments.get("segments", [])
    if not rows:
        return None
    values = rows[0].get("predictions", {}).get(key, [])
    if not values:
        return None
    return values[0]


def compact_item(item: dict) -> dict:
    return {
        "family": item.get("family"),
        "label": item.get("label"),
        "score": item.get("score"),
        "prompt": item.get("prompt"),
    }


def relative_strength(item: dict | None, primary_score: float) -> str:
    if not item:
        return "absent"
    score = float(item.get("score", 0.0))
    if primary_score <= 0:
        return "weak"
    ratio = score / primary_score
    if ratio >= 0.65:
        return "primary"
    if ratio >= 0.15:
        return "secondary"
    if ratio >= 0.02:
        return "weak"
    return "trace"


def candidate_layers(items: list[dict], reference_score: float, limit: int = 5, min_score: float = 0.0) -> tuple[list[dict], int]:
    layers = []
    seen = set()
    suppressed = 0
    for item in items:
        label = item.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        if float(item.get("score", 0.0)) < min_score:
            suppressed += 1
            continue
        strength = relative_strength(item, reference_score)
        if strength not in {"primary", "secondary", "weak"}:
            suppressed += 1
            continue
        layers.append(
            {
                "family": item.get("family"),
                "label": label,
                "strength": strength,
                "score": item.get("score"),
                "evidence_prompt": item.get("prompt"),
            }
        )
        if len(layers) >= limit:
            break
    return layers, suppressed


def build_profile(palette: dict, segments: dict | None, title: str) -> dict:
    ranked = top_overall(palette, 12)
    source_ranked = top_across_families(palette, SOURCE_FAMILIES, 12)
    fx_ranked = top_across_families(palette, {"fx_texture"}, 8)
    processing_ranked = top_across_families(palette, {"processing_space"}, 8)
    primary = source_ranked[0] if source_ranked else (ranked[0] if ranked else {})
    primary_score = float(primary.get("score", 0.0))
    primary_fx = fx_ranked[0] if fx_ranked and float(fx_ranked[0].get("score", 0.0)) >= MIN_FX_TEXTURE_SCORE else {}
    primary_processing = (
        processing_ranked[0]
        if processing_ranked and float(processing_ranked[0].get("score", 0.0)) >= MIN_PROCESSING_SCORE
        else {}
    )
    primary_fx_score = float(primary_fx.get("score", 0.0))
    primary_processing_score = float(primary_processing.get("score", 0.0))

    source_layers, suppressed_source = candidate_layers(source_ranked, primary_score, limit=6, min_score=MIN_SOURCE_SCORE)
    fx_texture_layers, suppressed_fx = candidate_layers(fx_ranked, primary_fx_score, limit=4, min_score=MIN_FX_TEXTURE_SCORE)
    processing_layers, suppressed_processing = candidate_layers(
        processing_ranked,
        primary_processing_score,
        limit=6,
        min_score=MIN_PROCESSING_SCORE,
    )
    processing = [
        {
            "label": layer["label"],
            "strength": layer["strength"],
            "score": layer["score"],
            "evidence_prompt": layer["evidence_prompt"],
        }
        for layer in processing_layers
    ]

    rf = {}
    if segments:
        for key in ["source_family", "reverb", "distortion", "filter_presence", "filter_motion_type", "stereo", "motion_presence"]:
            value = rf_top(segments, key)
            if value:
                rf[key] = value

    decision_notes = []
    if primary:
        decision_notes.append(f"primary_palette={primary.get('label')}")
    if rf.get("stereo", {}).get("label") == "wide":
        decision_notes.append("rf_dsp_supports_wide_stereo")
    if rf.get("motion_presence", {}).get("label") == "motion":
        decision_notes.append("rf_dsp_supports_temporal_motion")
    if rf.get("distortion", {}).get("confidence", 0.0) < 0.65:
        decision_notes.append("distortion_not_reliable_as_hard_label")

    return {
        "title": title,
        "audio": palette.get("audio"),
        "mode": "open_vocabulary_sound_palette_with_rf_dsp_crosscheck",
        "primary_sound": compact_item(primary) if primary else None,
        "primary_fx_texture": compact_item(primary_fx) if primary_fx else None,
        "primary_processing": compact_item(primary_processing) if primary_processing else None,
        "primary_palette_overall": compact_item(ranked[0]) if ranked else None,
        "source_layers": source_layers,
        "fx_texture_layers": fx_texture_layers,
        "likely_layers": source_layers + fx_texture_layers,
        "processing_cues": processing,
        "suppressed_low_confidence": {
            "source_layers": suppressed_source,
            "fx_texture_layers": suppressed_fx,
            "processing_cues": suppressed_processing,
        },
        "score_thresholds": {
            "source_layers": MIN_SOURCE_SCORE,
            "fx_texture_layers": MIN_FX_TEXTURE_SCORE,
            "processing_cues": MIN_PROCESSING_SCORE,
        },
        "ranked_palette": [compact_item(item) for item in ranked],
        "rf_dsp_crosscheck": rf,
        "decision_notes": decision_notes,
        "not_supported": [
            "exact VST plugin identification",
            "preset identification",
            "oscillator knob value recovery",
            "effect-chain order recovery",
            "true source separation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--palette", type=Path, required=True)
    parser.add_argument("--segments", type=Path, default=None)
    parser.add_argument("--title", default="External Clip")
    parser.add_argument("--out", type=Path, default=Path("outputs/sound_profile.json"))
    args = parser.parse_args()

    palette = load_json(args.palette)
    segments = load_json(args.segments) if args.segments else None
    profile = build_profile(palette, segments, args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
