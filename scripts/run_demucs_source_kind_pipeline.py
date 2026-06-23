from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def pick_model(preferred: Path) -> Path:
    if preferred.exists():
        return preferred
    fallback = Path("models/source_kind_multilabel_v1.joblib")
    if fallback.exists():
        return fallback
    raise SystemExit(f"No source-kind model found. Missing {preferred} and {fallback}.")


def write_index(out_root: Path, model: Path, demucs_model: str, threshold_scale: float) -> None:
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Demucs Stem Source-Kind Pipeline</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
li {{ margin: 10px 0; }}
.note {{ color: #444; max-width: 920px; }}
code {{ background: #f3f3f3; padding: 2px 4px; }}
</style>
<h1>Demucs Stem Source-Kind Pipeline</h1>
<p class="note">Recommended project path: split mixed clips into Demucs stems, analyze each stem in 2-second windows, then show a stem-aware source-kind timeline. This is more realistic than treating dense mixed clips as one blob.</p>
<p><b>Source-kind model:</b> <code>{model.as_posix()}</code><br><b>Demucs model:</b> <code>{demucs_model}</code><br><b>Threshold scale:</b> <code>{threshold_scale}</code></p>
<ul>
<li><a href="collection_summary.html">Collection Summary</a> - fastest overview by track and stem.</li>
<li><a href="stem_timeline_matrix.html">Stem Timeline Matrix</a> - main report: time rows by vocals/drums/bass/other stems.</li>
<li><a href="stem_source_kind.html">Stem Source-Kind Detail</a> - detailed row-level scores and stem audio controls.</li>
<li><a href="synth_focus.html">Synth Focus</a> - pulls synth-like candidates from all stems, because Demucs has no dedicated synth stem.</li>
<li><a href="synth_specialist_summary.html">Synth Specialist Summary</a> - compact track/region view from the focused synth model.</li>
<li><a href="synth_specialist.html">Synth Specialist</a> - focused synth/not-synth and synth-type model over every stem segment.</li>
<li><a href="synth_fast_ensemble.html">Fast Synth Ensemble</a> - practical agreement filter between synth specialist and source-kind stem model.</li>
<li><a href="synth_fast_ensemble_v2w_strict.html">Fast Synth Ensemble V2W Strict</a> - experimental pseudo-real weighted synth model with stricter agreement filtering, if generated.</li>
<li><a href="synth_fast_ensemble_v3_strict.html">Fast Synth Ensemble V3 Strict</a> - targeted synthetic rebalance experiment, if generated.</li>
<li><a href="source_kind_gap_audit.html">Source-Kind Gap Audit</a> - coverage and weak-label priorities.</li>
</ul>
<p class="note">For finer separation, rerun with <code>--demucs-model htdemucs_6s</code>. If available, that model adds <code>guitar</code> and <code>piano</code> stems between <code>bass</code> and <code>other</code>.</p>
<p class="note">Caution: Demucs separation artifacts and weak/self-trained source-kind labels can still produce false positives. Treat outputs as candidate layer hypotheses, not exact plugin/stem truth.</p>
</html>"""
    (out_root / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/demucs_source_kind_current"))
    parser.add_argument("--model", type=Path, default=Path("models/source_kind_multilabel_v2.joblib"))
    parser.add_argument("--demucs-model", default="htdemucs")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=2.0)
    parser.add_argument("--threshold-scale", type=float, default=0.85)
    parser.add_argument("--quality", choices=["fast", "librosa"], default="librosa")
    parser.add_argument("--synth-specialist-model", type=Path, default=Path("models/synth_specialist_v1.joblib"))
    parser.add_argument("--skip-synth-specialist", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-separation", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    model = pick_model(args.model)
    stems_root = args.out_root / args.demucs_model
    stem_csv = args.out_root / "stem_source_kind.csv"
    stem_html = args.out_root / "stem_source_kind.html"
    matrix_html = args.out_root / "stem_timeline_matrix.html"
    collection_csv = args.out_root / "collection_summary.csv"
    collection_html = args.out_root / "collection_summary.html"
    synth_csv = args.out_root / "synth_focus.csv"
    synth_html = args.out_root / "synth_focus.html"
    synth_specialist_csv = args.out_root / "synth_specialist.csv"
    synth_specialist_html = args.out_root / "synth_specialist.html"
    synth_specialist_track_csv = args.out_root / "synth_specialist_track_summary.csv"
    synth_specialist_region_csv = args.out_root / "synth_specialist_regions.csv"
    synth_specialist_summary_html = args.out_root / "synth_specialist_summary.html"
    synth_fast_ensemble_csv = args.out_root / "synth_fast_ensemble.csv"
    synth_fast_ensemble_html = args.out_root / "synth_fast_ensemble.html"
    audit_csv = args.out_root / "source_kind_gap_audit.csv"
    audit_html = args.out_root / "source_kind_gap_audit.html"

    args.out_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_separation:
        run(
            [
                py,
                "scripts/separate_external_loops_demucs.py",
                "--input-dir",
                str(args.input_dir),
                "--out-root",
                str(args.out_root),
                "--model",
                args.demucs_model,
                "--limit",
                "0",
                "--jobs",
                str(args.jobs),
                "--skip-existing",
                "--continue-on-error",
                "--sanitize-names",
            ]
        )

    run(
        [
            py,
            "scripts/analyze_demucs_stems_source_kind.py",
            "--stems-root",
            str(stems_root),
            "--model",
            str(model),
            "--segment-seconds",
            str(args.segment_seconds),
            "--hop-seconds",
            str(args.hop_seconds),
            "--quality",
            args.quality,
            "--strict-stem-kind",
            "--threshold-scale",
            str(args.threshold_scale),
            "--out-csv",
            str(stem_csv),
            "--out-html",
            str(stem_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_demucs_stem_timeline_matrix.py",
            "--input",
            str(stem_csv),
            "--out-html",
            str(matrix_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_demucs_collection_summary.py",
            "--input",
            str(stem_csv),
            "--out-csv",
            str(collection_csv),
            "--out-html",
            str(collection_html),
        ]
    )
    run(
        [
            py,
            "scripts/render_demucs_synth_focus.py",
            "--input",
            str(stem_csv),
            "--out-csv",
            str(synth_csv),
            "--out-html",
            str(synth_html),
        ]
    )
    if not args.skip_synth_specialist and args.synth_specialist_model.exists():
        run(
            [
                py,
                "scripts/analyze_demucs_synth_specialist.py",
                "--stems-root",
                str(stems_root),
                "--model",
                str(args.synth_specialist_model),
                "--segment-seconds",
                str(args.segment_seconds),
                "--hop-seconds",
                str(args.hop_seconds),
                "--quality",
                "librosa",
                "--out-csv",
                str(synth_specialist_csv),
                "--out-html",
                str(synth_specialist_html),
            ]
        )
        run(
            [
                py,
                "scripts/render_synth_specialist_summary.py",
                "--input",
                str(synth_specialist_csv),
                "--out-track-csv",
                str(synth_specialist_track_csv),
                "--out-region-csv",
                str(synth_specialist_region_csv),
                "--out-html",
                str(synth_specialist_summary_html),
            ]
        )
        run(
            [
                py,
                "scripts/ensemble_synth_fast.py",
                "--synth",
                str(synth_specialist_csv),
                "--source-kind",
                str(stem_csv),
                "--out-csv",
                str(synth_fast_ensemble_csv),
                "--out-html",
                str(synth_fast_ensemble_html),
            ]
        )
    run(
        [
            py,
            "scripts/audit_demucs_source_kind_gaps.py",
            "--stem-csv",
            str(stem_csv),
            "--out-csv",
            str(audit_csv),
            "--out-html",
            str(audit_html),
        ]
    )
    write_index(args.out_root, model, args.demucs_model, args.threshold_scale)
    print(f"\nMain report: {args.out_root / 'index.html'}")


if __name__ == "__main__":
    main()
