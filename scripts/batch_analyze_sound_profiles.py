from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from clap_palette_score import PaletteScorer
from create_sound_profile_json import build_profile


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sound_profiles"))
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--palette", type=Path, default=Path("configs/sound_palette_prompts.json"))
    parser.add_argument("--clap-model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--refresh-palette", action="store_true", help="Recompute CLAP palette JSON even when cached files exist.")
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    palette_scorer = None
    profile_paths = []
    for path in files:
        name = safe_stem(path)
        segments_json = args.out_dir / f"{name}_segments.json"
        palette_json = args.out_dir / f"{name}_palette.json"
        profile_json = args.out_dir / f"{name}_sound_profile.json"
        if not segments_json.exists():
            run(
                [
                    sys.executable,
                    "scripts/infer_audio_segments.py",
                    "--model",
                    str(args.model),
                    "--audio",
                    str(path),
                    "--out",
                    str(segments_json),
                ]
            )
        if palette_json.exists() and not args.refresh_palette:
            palette_report = json.loads(palette_json.read_text(encoding="utf-8"))
        else:
            if palette_scorer is None:
                palette_scorer = PaletteScorer(args.palette, args.clap_model_name)
            palette_report = palette_scorer.score(path)
            palette_json.write_text(json.dumps(palette_report, ensure_ascii=False, indent=2), encoding="utf-8")
        segments_report = json.loads(segments_json.read_text(encoding="utf-8"))
        profile = build_profile(palette_report, segments_report, name)
        profile_json.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"profile: {profile_json}")
        profile_paths.append(profile_json)

    primary_counts: Counter[str] = Counter()
    source_layer_counts: Counter[str] = Counter()
    fx_texture_counts: Counter[str] = Counter()
    processing_counts: Counter[str] = Counter()
    rows = []
    for profile_path in profile_paths:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        primary = profile.get("primary_sound") or {}
        primary_label = primary.get("label", "unknown")
        primary_counts[primary_label] += 1
        source_layers = profile.get("source_layers")
        if source_layers is None:
            source_layers = [
                layer
                for layer in profile.get("likely_layers", [])
                if layer.get("family") != "processing_space"
            ]
        fx_layers = profile.get("fx_texture_layers", [])
        for layer in source_layers:
            source_layer_counts[layer.get("label", "unknown")] += 1
        for layer in fx_layers:
            fx_texture_counts[layer.get("label", "unknown")] += 1
        for cue in profile.get("processing_cues", []):
            processing_counts[cue.get("label", "unknown")] += 1
        rows.append(
            {
                "file": profile_path.name,
                "primary": primary_label,
                "primary_score": primary.get("score", ""),
                "source_layers": ", ".join(layer.get("label", "") for layer in source_layers[:4]),
                "fx_textures": ", ".join(layer.get("label", "") for layer in fx_layers[:4]),
                "processing": ", ".join(cue.get("label", "") for cue in profile.get("processing_cues", [])[:4]),
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "profiles": [str(path) for path in profile_paths],
        "primary_counts": dict(primary_counts.most_common()),
        "source_layer_counts": dict(source_layer_counts.most_common()),
        "fx_texture_counts": dict(fx_texture_counts.most_common()),
        "processing_counts": dict(processing_counts.most_common()),
        "rows": rows,
    }
    summary_path = args.out_dir / "sound_profile_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def counter_table(title: str, counter: Counter[str]) -> str:
        body = "\n".join(f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in counter.most_common())
        return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"

    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr>"
            f"<td>{row['file']}</td><td>{row['primary']}</td><td>{row['primary_score']}</td>"
            f"<td>{row['source_layers']}</td><td>{row['fx_textures']}</td><td>{row['processing']}</td>"
            "</tr>"
        )
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Sound Profile Batch Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>Sound Profile Batch Summary</h1>",
            f"<p>Profiles: <strong>{len(profile_paths)}</strong></p>",
            "<div class='grid'>",
            counter_table("Primary Sound", primary_counts),
            counter_table("Source Layers", source_layer_counts),
            counter_table("FX / Texture Layers", fx_texture_counts),
            counter_table("Processing Cues", processing_counts),
            "</div>",
            "<h2>Details</h2>",
            "<table><thead><tr><th>Profile</th><th>Primary</th><th>Score</th><th>Source Layers</th><th>FX / Texture</th><th>Processing</th></tr></thead><tbody>",
            *detail_rows,
            "</tbody></table>",
        ]
    )
    html_path = args.out_dir / "sound_profile_batch_summary.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"summary: {html_path}")


if __name__ == "__main__":
    main()
