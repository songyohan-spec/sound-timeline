from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from analyze_sound_palette_timeline import analyze_timeline, render_html
from clap_palette_score import PaletteScorer
from dsp_palette_score import DSPPaletteScorer


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def counter_table(title: str, counter: Counter[str]) -> str:
    body = "\n".join(f"<tr><td>{label}</td><td>{count}</td></tr>" for label, count in counter.most_common())
    return f"<h2>{title}</h2><table><thead><tr><th>Label</th><th>Count</th></tr></thead><tbody>{body}</tbody></table>"


def first_label(item: dict | None) -> str:
    if not item:
        return "none"
    return str(item.get("label") or "none")


def analyze_one_file(task: tuple[str, str, str, float, float, str, str]) -> str:
    audio_path_s, out_json_s, out_html_s, segment_seconds, hop_seconds, palette_s, dsp_quality = task
    audio_path = Path(audio_path_s)
    out_json = Path(out_json_s)
    out_html = Path(out_html_s)
    name = safe_stem(audio_path)
    scorer = DSPPaletteScorer(Path(palette_s), quality=dsp_quality)
    report = analyze_timeline(audio_path, name, scorer, segment_seconds, hop_seconds)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_html.write_text(render_html(report), encoding="utf-8")
    return str(out_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/palette_timelines"))
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--palette", type=Path, default=Path("configs/sound_palette_prompts.json"))
    parser.add_argument("--clap-model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--scorer", choices=["clap", "dsp"], default="clap")
    parser.add_argument("--dsp-quality", choices=["librosa", "fast"], default="librosa")
    parser.add_argument("--workers", type=int, default=1, help="Parallel file workers for --scorer dsp.")
    parser.add_argument("--resume", action="store_true", help="Skip files whose timeline JSON and HTML already exist.")
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    timeline_paths = []
    pending_tasks = []
    for audio_path in files:
        name = safe_stem(audio_path)
        out_json = args.out_dir / f"{name}_palette_timeline.json"
        out_html = args.out_dir / f"{name}_palette_timeline.html"
        if args.resume and out_json.exists() and out_html.exists():
            print(f"skip existing: {audio_path}")
            timeline_paths.append(out_json)
            continue
        if args.scorer == "dsp" and args.workers > 1:
            pending_tasks.append((str(audio_path), str(out_json), str(out_html), args.segment_seconds, args.hop_seconds, str(args.palette), args.dsp_quality))
        else:
            scorer = PaletteScorer(args.palette, args.clap_model_name) if args.scorer == "clap" else DSPPaletteScorer(args.palette, quality=args.dsp_quality)
            print(f"analyzing: {audio_path}")
            report = analyze_timeline(audio_path, name, scorer, args.segment_seconds, args.hop_seconds)
            out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            out_html.write_text(render_html(report), encoding="utf-8")
            print(f"wrote: {out_json}")
            print(f"wrote: {out_html}")
            timeline_paths.append(out_json)

    if pending_tasks:
        print(f"parallel dsp tasks: {len(pending_tasks)} with workers={args.workers}")
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {executor.submit(analyze_one_file, task): task for task in pending_tasks}
            for future in as_completed(future_to_task):
                out_json = Path(future.result())
                print(f"wrote: {out_json}")
                timeline_paths.append(out_json)

    region_source_counts: Counter[str] = Counter()
    region_source_family_counts: Counter[str] = Counter()
    region_processing_counts: Counter[str] = Counter()
    segment_source_counts: Counter[str] = Counter()
    segment_source_family_counts: Counter[str] = Counter()
    segment_processing_counts: Counter[str] = Counter()
    source_layer_counts: Counter[str] = Counter()
    processing_cue_counts: Counter[str] = Counter()
    rows = []

    for path in timeline_paths:
        timeline = json.loads(path.read_text(encoding="utf-8"))
        segments_by_index = {
            segment.get("index"): segment
            for segment in timeline.get("segments", [])
        }
        for region in timeline.get("regions", []):
            region_source_counts[region.get("primary_source", "none")] += 1
            region_processing_counts[region.get("primary_processing", "none")] += 1
            first_segment = segments_by_index.get((region.get("segments") or [None])[0], {})
            primary_sound = first_segment.get("profile", {}).get("primary_sound") or {}
            region_source_family_counts[primary_sound.get("family", "none")] += 1
            rows.append(
                {
                    "file": path.name,
                    "time": f"{region.get('start'):.2f}-{region.get('end'):.2f}s",
                    "source": region.get("primary_source", "none"),
                    "source_family": primary_sound.get("family", "none"),
                    "processing": region.get("primary_processing", "none"),
                    "segments": ", ".join(str(index) for index in region.get("segments", [])),
                }
            )
        for segment in timeline.get("segments", []):
            profile = segment.get("profile", {})
            primary_sound = profile.get("primary_sound")
            primary_source_label = first_label(primary_sound)
            primary_source_family = (primary_sound or {}).get("family", "none")
            segment_source_counts[primary_source_label] += 1
            segment_source_family_counts[primary_source_family] += 1
            segment_processing_counts[first_label(profile.get("primary_processing"))] += 1
            for layer in profile.get("source_layers", []):
                source_layer_counts[layer.get("label", "none")] += 1
            for cue in profile.get("processing_cues", []):
                processing_cue_counts[cue.get("label", "none")] += 1

    summary = {
        "input_dir": str(args.input_dir),
        "timelines": [str(path) for path in timeline_paths],
        "region_source_counts": dict(region_source_counts.most_common()),
        "region_source_family_counts": dict(region_source_family_counts.most_common()),
        "region_processing_counts": dict(region_processing_counts.most_common()),
        "segment_source_counts": dict(segment_source_counts.most_common()),
        "segment_source_family_counts": dict(segment_source_family_counts.most_common()),
        "segment_processing_counts": dict(segment_processing_counts.most_common()),
        "source_layer_counts": dict(source_layer_counts.most_common()),
        "processing_cue_counts": dict(processing_cue_counts.most_common()),
        "rows": rows,
    }
    summary_json = args.out_dir / "palette_timeline_batch_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_rows = "\n".join(
        "<tr>"
        f"<td>{row['file']}</td><td>{row['time']}</td><td>{row['source_family']}</td><td>{row['source']}</td>"
        f"<td>{row['processing']}</td><td>{row['segments']}</td>"
        "</tr>"
        for row in rows
    )
    html = "\n".join(
        [
            "<!doctype html>",
            "<meta charset='utf-8'>",
            "<title>Palette Timeline Batch Summary</title>",
            "<style>body{font-family:Arial,sans-serif;max-width:1200px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%;margin:12px 0 24px}th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}th{background:#f2f2f2}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}</style>",
            "<h1>Palette Timeline Batch Summary</h1>",
            f"<p>Files: <strong>{len(timeline_paths)}</strong></p>",
            "<div class='grid'>",
            counter_table("Region Primary Source", region_source_counts),
            counter_table("Segment Primary Source Family", segment_source_family_counts),
            counter_table("Region Primary Processing", region_processing_counts),
            counter_table("Segment Primary Source", segment_source_counts),
            counter_table("Segment Primary Processing", segment_processing_counts),
            counter_table("Source Layer Mentions", source_layer_counts),
            counter_table("Processing Cue Mentions", processing_cue_counts),
            "</div>",
            "<h2>Regions</h2>",
            "<table><thead><tr><th>Timeline</th><th>Time</th><th>Source Family</th><th>Primary Source</th><th>Primary Processing</th><th>Segments</th></tr></thead><tbody>",
            detail_rows,
            "</tbody></table>",
        ]
    )
    summary_html = args.out_dir / "palette_timeline_batch_summary.html"
    summary_html.write_text(html, encoding="utf-8")
    print(f"summary: {summary_json}")
    print(f"summary: {summary_html}")


if __name__ == "__main__":
    main()
