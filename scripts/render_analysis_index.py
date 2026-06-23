from __future__ import annotations

import argparse
from pathlib import Path


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/open_source_panel_current"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/open_source_panel_current/index.html"))
    args = parser.parse_args()

    root = args.root
    links = [
        ("Source Kind Detail", root / "all_segments_queue" / "source_kind_detail.html", "Best current view for mid-level source kinds such as vocal chop, synth pad, 808/sub, drum groove, guitar loop, sampled texture, and FX/noise."),
        ("AST Source Kind Teacher", root / "all_segments_queue" / "ast_source_kind_teacher.html", "Open-source AST/AudioSet-only source-kind mapping, separated from project-trained model guesses."),
        ("Source Kind Model Check", root / "all_segments_queue" / "source_kind_model_check.html", "Compares heuristic/external-panel source-kind candidates against the trained source-kind model."),
        ("Source Kind Training Coverage", root / "all_segments_queue" / "source_kind_training_coverage.html", "Checks which source-kind candidates are actually backed by training folders/examples and which need data next."),
        ("Layer Matrix", root / "all_segments_queue" / "sound_layer_matrix.html", "Best current view for co-occurring layer candidates by time."),
        ("Collection Overview", root / "all_segments_queue" / "collection_overview.html", "Best current view for song/prefix-level palette differences across the dataset."),
        ("Sound Element Timeline", root / "all_segments_queue" / "sound_element_timeline.html", "Best current view for primary sound read per 2-second segment."),
        ("Vocal / Synth Detail", root / "all_segments_queue" / "vocal_synth_detail.html", "Best current view for processed vocal and synth-family hypotheses."),
        ("Rhythm Section Detail", root / "all_segments_queue" / "rhythm_section_detail.html", "Best current view for kick, 808/sub, snare/clap, glitch percussion, and pumping cues."),
        ("Model Diagnostics", root / "all_segments_queue" / "model_diagnostics.html", "Best current view for repeated model failure modes and improvement priorities."),
        ("Clip Sound Cue Summary", root / "all_segments_queue" / "clip_sound_cue_summary.html", "Per-file summary of public cues and project candidates."),
        ("Public Cue Translation", root / "all_segments_queue" / "audioset_sound_cues.html", "Raw public AudioSet tags translated into sound-design cues."),
        ("Public Model Filter", root / "all_segments_queue" / "public_model_filtered.html", "Where the project model is supported or demoted by AST/AudioSet."),
        ("Review Queue", root / "review_queue" / "open_source_panel_report.html", "Top 30 high-priority review rows."),
        ("Reference Ensemble", root / "reference_ensemble.html", "Project model synthetic-reference outputs before public filtering."),
    ]
    items = []
    for title, path, desc in links:
        if path.exists():
            href = rel(path, root)
            items.append(f"<li><a href='{href}'>{title}</a><br><span>{desc}</span></li>")
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Sound Analysis Index</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #111; max-width: 920px; }}
li {{ margin: 16px 0; }}
a {{ font-size: 20px; font-weight: bold; color: #164a9b; }}
span {{ color: #555; }}
</style>
<h1>Sound Analysis Index</h1>
<p>Open the reports in this order. The first two are the most useful for listening and inspection.</p>
<ol>{''.join(items)}</ol>
</html>"""
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(page, encoding="utf-8")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
