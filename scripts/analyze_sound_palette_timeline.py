from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from clap_palette_score import PaletteScorer
from create_sound_profile_json import build_profile
from dsp_palette_score import DSPPaletteScorer


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def load_timeline_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32), sr


def describe_items(items: list[dict], limit: int) -> str:
    parts = []
    for item in items[:limit]:
        score = item.get("score")
        score_text = f"{float(score):.3f}" if isinstance(score, (int, float)) else str(score)
        parts.append(f"{item.get('label', '')} ({item.get('strength', '')}, {score_text})")
    return "<br>".join(parts)


def label_of(item: dict | None) -> str:
    if not item:
        return "none"
    return str(item.get("label") or "none")


def build_regions(segments: list[dict]) -> list[dict]:
    regions: list[dict] = []
    for segment in segments:
        profile = segment["profile"]
        source = label_of(profile.get("primary_sound"))
        processing = label_of(profile.get("primary_processing"))
        key = (source, processing)
        if regions and regions[-1]["key"] == key and abs(regions[-1]["end"] - segment["start"]) < 0.001:
            regions[-1]["end"] = segment["end"]
            regions[-1]["segments"].append(segment["index"])
            continue
        regions.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "key": key,
                "primary_source": source,
                "primary_processing": processing,
                "segments": [segment["index"]],
            }
        )
    for region in regions:
        region.pop("key", None)
    return regions


def render_html(report: dict) -> str:
    region_rows = []
    for region in report.get("regions", []):
        region_rows.append(
            "<tr>"
            f"<td>{region['start']:.2f}-{region['end']:.2f}s</td>"
            f"<td>{region['primary_source']}</td>"
            f"<td>{region['primary_processing']}</td>"
            f"<td>{', '.join(str(index) for index in region['segments'])}</td>"
            "</tr>"
        )

    rows = []
    for segment in report["segments"]:
        profile = segment["profile"]
        primary_source = profile.get("primary_sound") or {}
        primary_processing = profile.get("primary_processing") or {}
        source_layers = describe_items(profile.get("source_layers", []), 5)
        fx_layers = describe_items(profile.get("fx_texture_layers", []), 3)
        processing = describe_items(profile.get("processing_cues", []), 5)
        suppressed = profile.get("suppressed_low_confidence", {})
        suppressed_text = ", ".join(
            f"{key}:{value}"
            for key, value in suppressed.items()
            if value
        )
        rows.append(
            "<tr>"
            f"<td>{segment['start']:.2f}-{segment['end']:.2f}s</td>"
            f"<td>{primary_source.get('label', 'none')}</td>"
            f"<td>{primary_source.get('score', '')}</td>"
            f"<td>{primary_processing.get('label', 'none')}</td>"
            f"<td>{primary_processing.get('score', '')}</td>"
            f"<td>{source_layers}</td>"
            f"<td>{fx_layers}</td>"
            f"<td>{processing}</td>"
            f"<td>{suppressed_text}</td>"
            "</tr>"
        )
    return "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            f"<title>{report['title']} Sound Palette Timeline</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.note{color:#555}</style>",
            f"<h1>{report['title']} Sound Palette Timeline</h1>",
            f"<p class='note'>Audio: {report['audio']}</p>",
            "<h2>Region Summary</h2>",
            "<table><thead><tr><th>Time</th><th>Primary Source</th><th>Primary Processing</th><th>Segments</th></tr></thead><tbody>",
            *region_rows,
            "</tbody></table>",
            "<h2>Segment Detail</h2>",
            "<table><thead><tr><th>Time</th><th>Primary Source</th><th>Source Score</th><th>Primary Processing</th><th>Processing Score</th><th>Source Layers</th><th>FX / Texture</th><th>Processing Cues</th><th>Suppressed</th></tr></thead><tbody>",
            *rows,
            "</tbody></table>",
            "<p class='note'>CLAP palette rankings are semantic hints, not exact source separation or plugin recovery.</p>",
        ]
    )


def analyze_timeline(
    audio_path: Path,
    title: str,
    scorer: PaletteScorer,
    segment_seconds: float,
    hop_seconds: float,
) -> dict:
    audio, sr = load_timeline_audio(audio_path)
    seg_len = int(segment_seconds * sr)
    hop_len = int(hop_seconds * sr)
    if len(audio) < seg_len:
        audio = np.pad(audio, ((0, seg_len - len(audio)), (0, 0)))

    segments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        start = 0
        index = 0
        while start + seg_len <= len(audio):
            end = start + seg_len
            segment_audio = audio[start:end]
            segment_path = tmp_dir / f"segment_{index:04d}.wav"
            sf.write(segment_path, segment_audio, sr)
            palette_report = scorer.score(segment_path)
            profile = build_profile(palette_report, None, f"{title}_{index:04d}")
            segments.append(
                {
                    "index": index,
                    "start": round(start / sr, 3),
                    "end": round(end / sr, 3),
                    "profile": profile,
                }
            )
            index += 1
            start += hop_len

    return {
        "title": title,
        "audio": str(audio_path),
        "segment_seconds": segment_seconds,
        "hop_seconds": hop_seconds,
        "mode": "open_vocabulary_sound_palette_timeline",
        "regions": build_regions(segments),
        "segments": segments,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--palette", type=Path, default=Path("configs/sound_palette_prompts.json"))
    parser.add_argument("--clap-model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--scorer", choices=["clap", "dsp"], default="clap")
    parser.add_argument("--dsp-quality", choices=["librosa", "fast"], default="librosa")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    args = parser.parse_args()

    title = args.name or safe_stem(args.audio)
    out_json = args.out_json or Path("outputs") / f"{title}_palette_timeline.json"
    out_html = args.out_html or Path("outputs") / f"{title}_palette_timeline.html"

    scorer = PaletteScorer(args.palette, args.clap_model_name) if args.scorer == "clap" else DSPPaletteScorer(args.palette, quality=args.dsp_quality)
    report = analyze_timeline(args.audio, title, scorer, args.segment_seconds, args.hop_seconds)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render_html(report), encoding="utf-8")
    print(f"wrote: {out_json}")
    print(f"wrote: {out_html}")


if __name__ == "__main__":
    main()
