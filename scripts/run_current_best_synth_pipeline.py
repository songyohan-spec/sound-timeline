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
    parser.add_argument("--out-root", type=Path, default=Path("outputs/demucs_stems_6s_full"))
    parser.add_argument("--feature-cache", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_feature_cache_2s_librosa.csv"))
    parser.add_argument("--synth-model", type=Path, default=Path("models/synth_specialist_v4.joblib"))
    parser.add_argument("--source-kind", type=Path, default=Path("outputs/demucs_stems_6s_full/stem_source_kind_merged_v3_cached_batch.csv"))
    parser.add_argument("--teacher-queue-limit", type=int, default=60)
    parser.add_argument("--teacher-queue-per-label", type=int, default=8)
    args = parser.parse_args()

    py = sys.executable
    out = args.out_root
    synth_csv = out / "synth_specialist_v4_cached_batch.csv"
    synth_html = out / "synth_specialist_v4_cached_batch.html"
    ensemble_csv = out / "synth_fast_ensemble_v4_cached_aligned_strict_texture.csv"
    ensemble_html = out / "synth_fast_ensemble_v4_cached_aligned_strict_texture.html"
    bottleneck_csv = out / "synth_evidence_bottlenecks_v4_strict_texture.csv"
    bottleneck_html = out / "synth_evidence_bottlenecks_v4_strict_texture.html"
    high_precision_csv = out / "synth_high_precision_v4_strict.csv"
    high_precision_regions = out / "synth_high_precision_v4_strict_regions.csv"
    high_precision_html = out / "synth_high_precision_v4_strict.html"
    broad_multilayer_csv = out / "broad_multilayer_timeline.csv"
    broad_multilayer_html = out / "broad_multilayer_timeline.html"
    broad_regions_likely_csv = out / "broad_layer_regions_likely.csv"
    broad_regions_likely_html = out / "broad_layer_regions_likely.html"
    broad_regions_strong_csv = out / "broad_layer_regions_strong.csv"
    broad_regions_strong_html = out / "broad_layer_regions_strong.html"
    broad_track_overview_csv = out / "broad_track_overview.csv"
    broad_track_overview_html = out / "broad_track_overview.html"
    feature_cache_4s = out / "stem_feature_cache_4s_hop2_librosa.csv"
    source_kind_4s_csv = out / "stem_source_kind_v3_targeted_4s_hop2.csv"
    source_kind_4s_html = out / "stem_source_kind_v3_targeted_4s_hop2.html"
    synth_4s_csv = out / "synth_specialist_v4_4s_hop2.csv"
    synth_4s_html = out / "synth_specialist_v4_4s_hop2.html"
    ensemble_4s_csv = out / "synth_fast_ensemble_v4_4s_hop2.csv"
    ensemble_4s_html = out / "synth_fast_ensemble_v4_4s_hop2.html"
    multiscale_csv = out / "multiscale_synth_overlay.csv"
    multiscale_html = out / "multiscale_synth_overlay.html"
    broad_multiscale_csv = out / "broad_multilayer_timeline_multiscale.csv"
    broad_multiscale_html = out / "broad_multilayer_timeline_multiscale.html"
    broad_regions_likely_multiscale_csv = out / "broad_layer_regions_likely_multiscale.csv"
    broad_regions_likely_multiscale_html = out / "broad_layer_regions_likely_multiscale.html"
    synth_candidate_audio_csv = out / "synth_candidate_audio_index.csv"
    synth_candidate_audio_html = out / "synth_candidate_audio_index.html"
    synth_candidate_audio_strict_csv = out / "synth_candidate_audio_strict_index.csv"
    synth_candidate_audio_strict_html = out / "synth_candidate_audio_strict_index.html"
    synth_candidate_audio_strict_triage_csv = out / "synth_candidate_audio_strict_triage.csv"
    synth_candidate_audio_strict_triage_html = out / "synth_candidate_audio_strict_triage.html"
    synth_candidate_audio_auditionable_csv = out / "synth_candidate_audio_auditionable.csv"
    synth_candidate_audio_auditionable_html = out / "synth_candidate_audio_auditionable.html"
    synth_candidate_audio_failure_csv = out / "synth_candidate_audio_failure_audit.csv"
    synth_candidate_audio_failure_html = out / "synth_candidate_audio_failure_audit.html"
    teacher_queue = out / "synth_teacher_queue_v4_strict.csv"
    teacher_queue_html = out / "synth_teacher_queue_v4_strict.html"

    if not args.feature_cache.exists():
        raise SystemExit(f"Missing feature cache: {args.feature_cache}")
    if not args.source_kind.exists():
        raise SystemExit(f"Missing merged source-kind evidence: {args.source_kind}")

    run(
        [
            py,
            "scripts/analyze_demucs_synth_specialist.py",
            "--model",
            str(args.synth_model),
            "--feature-cache",
            str(args.feature_cache),
            "--out-csv",
            str(synth_csv),
            "--out-html",
            str(synth_html),
        ]
    )
    run(
        [
            py,
            "scripts/ensemble_synth_fast.py",
            "--synth",
            str(synth_csv),
            "--source-kind",
            str(args.source_kind),
            "--out-csv",
            str(ensemble_csv),
            "--out-html",
            str(ensemble_html),
        ]
    )
    run(
        [
            py,
            "scripts/audit_synth_evidence_bottlenecks.py",
            "--input",
            str(ensemble_csv),
            "--out-csv",
            str(bottleneck_csv),
            "--out-html",
            str(bottleneck_html),
        ]
    )
    run(
        [
            py,
            "scripts/filter_synth_high_precision.py",
            "--ensemble",
            str(ensemble_csv),
            "--source-kind",
            str(args.source_kind),
            "--out-csv",
            str(high_precision_csv),
            "--out-regions",
            str(high_precision_regions),
            "--out-html",
            str(high_precision_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_multilayer_timeline.py",
            "--source-kind",
            str(args.source_kind),
            "--synth",
            str(ensemble_csv),
            "--high-precision",
            str(high_precision_csv),
            "--out-csv",
            str(broad_multilayer_csv),
            "--out-html",
            str(broad_multilayer_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_layer_regions.py",
            "--input",
            str(broad_multilayer_csv),
            "--min-strength",
            "likely",
            "--out-csv",
            str(broad_regions_likely_csv),
            "--out-html",
            str(broad_regions_likely_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_layer_regions.py",
            "--input",
            str(broad_multilayer_csv),
            "--min-strength",
            "strong",
            "--out-csv",
            str(broad_regions_strong_csv),
            "--out-html",
            str(broad_regions_strong_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_track_overview.py",
            "--likely",
            str(broad_regions_likely_csv),
            "--strong",
            str(broad_regions_strong_csv),
            "--out-csv",
            str(broad_track_overview_csv),
            "--out-html",
            str(broad_track_overview_html),
        ]
    )
    run(
        [
            py,
            "scripts/export_demucs_stem_feature_cache.py",
            "--stems-root",
            str(out / "htdemucs_6s"),
            "--segment-seconds",
            "4",
            "--hop-seconds",
            "2",
            "--quality",
            "librosa",
            "--workers",
            "4",
            "--out",
            str(feature_cache_4s),
        ]
    )
    run(
        [
            py,
            "scripts/analyze_demucs_stems_source_kind.py",
            "--model",
            "models/source_kind_multilabel_v3_targeted.joblib",
            "--feature-cache",
            str(feature_cache_4s),
            "--threshold-scale",
            "0.85",
            "--out-csv",
            str(source_kind_4s_csv),
            "--out-html",
            str(source_kind_4s_html),
        ]
    )
    run(
        [
            py,
            "scripts/analyze_demucs_synth_specialist.py",
            "--model",
            str(args.synth_model),
            "--feature-cache",
            str(feature_cache_4s),
            "--out-csv",
            str(synth_4s_csv),
            "--out-html",
            str(synth_4s_html),
        ]
    )
    run(
        [
            py,
            "scripts/ensemble_synth_fast.py",
            "--synth",
            str(synth_4s_csv),
            "--source-kind",
            str(source_kind_4s_csv),
            "--out-csv",
            str(ensemble_4s_csv),
            "--out-html",
            str(ensemble_4s_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_multiscale_synth_overlay.py",
            "--timeline-2s",
            str(broad_multilayer_csv),
            "--synth-4s",
            str(ensemble_4s_csv),
            "--out-csv",
            str(multiscale_csv),
            "--out-html",
            str(multiscale_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_multilayer_timeline_multiscale.py",
            "--broad",
            str(broad_multilayer_csv),
            "--multiscale",
            str(multiscale_csv),
            "--out-csv",
            str(broad_multiscale_csv),
            "--out-html",
            str(broad_multiscale_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_broad_layer_regions.py",
            "--input",
            str(broad_multiscale_csv),
            "--min-strength",
            "likely",
            "--use-synth-context",
            "--out-csv",
            str(broad_regions_likely_multiscale_csv),
            "--out-html",
            str(broad_regions_likely_multiscale_html),
        ]
    )
    run(
        [
            py,
            "scripts/export_synth_candidate_audio.py",
            "--regions",
            str(broad_regions_likely_multiscale_csv),
            "--stems-root",
            str(out / "htdemucs_6s"),
            "--out-root",
            str(out / "synth_candidate_audio"),
            "--out-csv",
            str(synth_candidate_audio_csv),
            "--out-html",
            str(synth_candidate_audio_html),
        ]
    )
    run(
        [
            py,
            "scripts/export_synth_candidate_audio.py",
            "--regions",
            str(broad_regions_likely_multiscale_csv),
            "--stems-root",
            str(out / "htdemucs_6s"),
            "--out-root",
            str(out / "synth_candidate_audio_strict"),
            "--no-include-bass",
            "--no-include-vocal-hybrid",
            "--out-csv",
            str(synth_candidate_audio_strict_csv),
            "--out-html",
            str(synth_candidate_audio_strict_html),
        ]
    )
    run(
        [
            py,
            "scripts/score_synth_candidate_separation.py",
            "--index",
            str(synth_candidate_audio_strict_csv),
            "--out-csv",
            str(synth_candidate_audio_strict_triage_csv),
            "--out-html",
            str(synth_candidate_audio_strict_triage_html),
        ]
    )
    run(
        [
            py,
            "scripts/filter_auditionable_synth_candidates.py",
            "--triage",
            str(synth_candidate_audio_strict_triage_csv),
            "--out-csv",
            str(synth_candidate_audio_auditionable_csv),
            "--out-html",
            str(synth_candidate_audio_auditionable_html),
        ]
    )
    run(
        [
            py,
            "scripts/audit_synth_candidate_failures.py",
            "--triage",
            str(synth_candidate_audio_strict_triage_csv),
            "--out-csv",
            str(synth_candidate_audio_failure_csv),
            "--out-html",
            str(synth_candidate_audio_failure_html),
        ]
    )
    run(
        [
            py,
            "scripts/select_synth_teacher_queue.py",
            "--synth",
            str(synth_csv),
            "--ensemble",
            str(ensemble_csv),
            "--limit",
            str(args.teacher_queue_limit),
            "--per-label",
            str(args.teacher_queue_per_label),
            "--out",
            str(teacher_queue),
        ]
    )
    run(
        [
            py,
            "scripts/render_synth_teacher_queue.py",
            "--input",
            str(teacher_queue),
            "--out-html",
            str(teacher_queue_html),
        ]
    )
    run(
        [
            py,
            "scripts/compare_synth_ensembles.py",
            f"current={ensemble_csv}",
        ]
    )
    print(f"\nCurrent conservative synth baseline: {ensemble_html}")
    print(f"Broad multi-layer timeline: {broad_multilayer_html}")
    print(f"Broad multi-layer timeline + 4s synth context: {broad_multiscale_html}")
    print(f"Broad likely regions + 4s synth context: {broad_regions_likely_multiscale_html}")
    print(f"Synth candidate audio exports: {synth_candidate_audio_html}")
    print(f"Strict synth candidate audio exports: {synth_candidate_audio_strict_html}")
    print(f"Strict synth separation triage: {synth_candidate_audio_strict_triage_html}")
    print(f"Auditionable strict synth candidates: {synth_candidate_audio_auditionable_html}")
    print(f"Strict synth candidate failure audit: {synth_candidate_audio_failure_html}")
    print(f"Broad likely/strong layer regions: {broad_regions_likely_html}")
    print(f"Broad strong-only layer regions: {broad_regions_strong_html}")
    print(f"Broad track overview: {broad_track_overview_html}")
    print(f"Multi-scale synth overlay: {multiscale_html}")
    print(f"4s synth ensemble: {ensemble_4s_html}")
    print(f"High precision synth shortlist: {high_precision_html}")
    print(f"Teacher review queue: {teacher_queue}")
    print(f"Teacher review queue HTML: {teacher_queue_html}")


if __name__ == "__main__":
    main()
