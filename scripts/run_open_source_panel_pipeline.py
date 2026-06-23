from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold-floor", type=float, default=0.45)
    parser.add_argument("--out-root", type=Path, default=Path("outputs/open_source_panel"))
    parser.add_argument("--all-segments", action="store_true", help="Also run the public panel over every segment, not only the review queue.")
    args = parser.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    batch_csv = out_root / "reference_ensemble.csv"
    batch_html = out_root / "reference_ensemble.html"
    batch_json = out_root / "reference_ensemble.json"
    review_dir = out_root / "review_queue"
    teacher_csv = review_dir / "audioset_teacher.csv"
    teacher_html = review_dir / "audioset_teacher.html"
    filtered_csv = review_dir / "public_model_filtered.csv"
    filtered_html = review_dir / "public_model_filtered.html"
    final_html = review_dir / "open_source_panel_report.html"
    cue_csv = review_dir / "audioset_sound_cues.csv"
    cue_html = review_dir / "audioset_sound_cues.html"

    py = sys.executable
    run(
        [
            py,
            "scripts/batch_infer_reference_ensemble.py",
            "--input-dir",
            str(args.input_dir),
            "--segment-seconds",
            str(args.segment_seconds),
            "--hop-seconds",
            str(args.hop_seconds),
            "--threshold-floor",
            str(args.threshold_floor),
            "--out-json",
            str(batch_json),
            "--out-csv",
            str(batch_csv),
            "--out-html",
            str(batch_html),
        ]
    )
    run(
        [
            py,
            "scripts/create_segment_review_queue.py",
            "--input",
            str(batch_csv),
            "--audio-root",
            str(args.input_dir),
            "--out-dir",
            str(review_dir),
            "--limit",
            str(args.limit),
            "--mode",
            "detected",
        ]
    )
    run(
        [
            py,
            "scripts/audioset_teacher_review_queue.py",
            "--queue",
            str(review_dir / "review_queue.csv"),
            "--limit",
            str(args.limit),
            "--out-csv",
            str(teacher_csv),
            "--out-html",
            str(teacher_html),
        ]
    )
    run(
        [
            py,
            "scripts/apply_audioset_teacher_filter.py",
            "--teacher",
            str(teacher_csv),
            "--out-csv",
            str(filtered_csv),
            "--out-html",
            str(filtered_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_public_sound_element_report.py",
            "--input",
            str(filtered_csv),
            "--out-html",
            str(final_html),
        ]
    )
    run(
        [
            py,
            "scripts/interpret_audioset_sound_cues.py",
            "--input",
            str(filtered_csv),
            "--out-csv",
            str(cue_csv),
            "--out-html",
            str(cue_html),
        ]
    )
    print(f"\nFinal report: {final_html}")
    print(f"Sound cue report: {cue_html}")

    if args.all_segments:
        all_dir = out_root / "all_segments_queue"
        all_teacher_csv = all_dir / "audioset_teacher.csv"
        all_filtered_csv = all_dir / "public_model_filtered.csv"
        all_cue_csv = all_dir / "audioset_sound_cues.csv"
        all_timeline_csv = all_dir / "sound_element_timeline.csv"
        all_timeline_html = all_dir / "sound_element_timeline.html"
        all_matrix_csv = all_dir / "sound_layer_matrix.csv"
        all_matrix_html = all_dir / "sound_layer_matrix.html"
        all_diag_html = all_dir / "model_diagnostics.html"
        all_rhythm_csv = all_dir / "rhythm_section_detail.csv"
        all_rhythm_html = all_dir / "rhythm_section_detail.html"
        all_vocal_synth_csv = all_dir / "vocal_synth_detail.csv"
        all_vocal_synth_html = all_dir / "vocal_synth_detail.html"
        all_source_kind_csv = all_dir / "source_kind_detail.csv"
        all_source_kind_html = all_dir / "source_kind_detail.html"
        all_ast_source_kind_csv = all_dir / "ast_source_kind_teacher.csv"
        all_ast_source_kind_html = all_dir / "ast_source_kind_teacher.html"
        all_source_kind_coverage_csv = all_dir / "source_kind_training_coverage.csv"
        all_source_kind_coverage_html = all_dir / "source_kind_training_coverage.html"
        all_source_kind_model_check_csv = all_dir / "source_kind_model_check.csv"
        all_source_kind_model_check_html = all_dir / "source_kind_model_check.html"
        all_collection_html = all_dir / "collection_overview.html"
        run(
            [
                py,
                "scripts/create_segment_review_queue.py",
                "--input",
                str(batch_csv),
                "--audio-root",
                str(args.input_dir),
                "--out-dir",
                str(all_dir),
                "--limit",
                "999",
                "--mode",
                "all",
            ]
        )
        run(
            [
                py,
                "scripts/audioset_teacher_review_queue.py",
                "--queue",
                str(all_dir / "review_queue.csv"),
                "--limit",
                "999",
                "--out-csv",
                str(all_teacher_csv),
                "--out-html",
                str(all_dir / "audioset_teacher.html"),
            ]
        )
        run(
            [
                py,
                "scripts/apply_audioset_teacher_filter.py",
                "--teacher",
                str(all_teacher_csv),
                "--out-csv",
                str(all_filtered_csv),
                "--out-html",
                str(all_dir / "public_model_filtered.html"),
            ]
        )
        run(
            [
                py,
                "scripts/interpret_audioset_sound_cues.py",
                "--input",
                str(all_filtered_csv),
                "--out-csv",
                str(all_cue_csv),
                "--out-html",
                str(all_dir / "audioset_sound_cues.html"),
            ]
        )
        run(
            [
                py,
                "scripts/render_sound_element_timeline_report.py",
                "--input",
                str(all_cue_csv),
                "--out-csv",
                str(all_timeline_csv),
                "--out-html",
                str(all_timeline_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_layer_matrix_report.py",
                "--input",
                str(all_cue_csv),
                "--timeline",
                str(all_timeline_csv),
                "--out-csv",
                str(all_matrix_csv),
                "--out-html",
                str(all_matrix_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_model_diagnostics.py",
                "--matrix",
                str(all_matrix_csv),
                "--timeline",
                str(all_timeline_csv),
                "--filtered",
                str(all_filtered_csv),
                "--out-html",
                str(all_diag_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_rhythm_section_report.py",
                "--input",
                str(all_cue_csv),
                "--out-csv",
                str(all_rhythm_csv),
                "--out-html",
                str(all_rhythm_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_vocal_synth_detail_report.py",
                "--input",
                str(all_cue_csv),
                "--out-csv",
                str(all_vocal_synth_csv),
                "--out-html",
                str(all_vocal_synth_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_source_kind_report.py",
                "--input",
                str(all_cue_csv),
                "--out-csv",
                str(all_source_kind_csv),
                "--out-html",
                str(all_source_kind_html),
            ]
        )
        run(
            [
                py,
                "scripts/ast_source_kind_teacher.py",
                "--input",
                str(all_cue_csv),
                "--heuristic",
                str(all_source_kind_csv),
                "--out-csv",
                str(all_ast_source_kind_csv),
                "--out-html",
                str(all_ast_source_kind_html),
            ]
        )
        run(
            [
                py,
                "scripts/audit_source_kind_training_coverage.py",
                "--source-kind-csv",
                str(all_source_kind_csv),
                "--out-csv",
                str(all_source_kind_coverage_csv),
                "--out-html",
                str(all_source_kind_coverage_html),
            ]
        )
        source_kind_model = Path("models/source_kind_multilabel_v1.joblib")
        if source_kind_model.exists():
            run(
                [
                    py,
                    "scripts/infer_source_kind_model_on_queue.py",
                    "--queue",
                    str(all_dir / "review_queue.csv"),
                    "--queue-root",
                    str(all_dir),
                    "--model",
                    str(source_kind_model),
                    "--heuristic",
                    str(all_source_kind_csv),
                    "--quality",
                    "librosa",
                    "--out-csv",
                    str(all_source_kind_model_check_csv),
                    "--out-html",
                    str(all_source_kind_model_check_html),
                ]
            )
        run(
            [
                py,
                "scripts/render_collection_overview.py",
                "--layer",
                str(all_matrix_csv),
                "--vocal-synth",
                str(all_vocal_synth_csv),
                "--rhythm",
                str(all_rhythm_csv),
                "--timeline",
                str(all_timeline_csv),
                "--out-html",
                str(all_collection_html),
            ]
        )
        print(f"Full segment timeline: {all_timeline_html}")
        print(f"Layer matrix: {all_matrix_html}")
        print(f"Diagnostics: {all_diag_html}")
        print(f"Rhythm section: {all_rhythm_html}")
        print(f"Vocal/synth detail: {all_vocal_synth_html}")
        print(f"Source kind detail: {all_source_kind_html}")
        print(f"AST source kind teacher: {all_ast_source_kind_html}")
        print(f"Source kind training coverage: {all_source_kind_coverage_html}")
        if all_source_kind_model_check_html.exists():
            print(f"Source kind model check: {all_source_kind_model_check_html}")
        print(f"Collection overview: {all_collection_html}")

    run(
        [
            py,
            "scripts/render_analysis_index.py",
            "--root",
            str(out_root),
            "--out-html",
            str(out_root / "index.html"),
        ]
    )
    print(f"Index: {out_root / 'index.html'}")


if __name__ == "__main__":
    main()
