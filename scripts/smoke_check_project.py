from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REQUIRED_SOURCE_FILES = [
    "README.md",
    "environment.yml",
    ".gitignore",
    "docs/next_build_plan.md",
    "docs/local_artifacts.md",
    "scripts/run_current_best_synth_pipeline.py",
    "scripts/export_synth_candidate_audio.py",
    "scripts/score_synth_candidate_separation.py",
    "scripts/render_broad_multilayer_timeline_multiscale.py",
    "src/sound_timeline/features.py",
    "src/sound_timeline/synthesis.py",
]

REQUIRED_CONFIG_FILES = [
    "configs/source_kind_training_map.json",
    "configs/synth_specialist_clap_prompts.json",
    "configs/sound_element_ontology.json",
    "configs/reference_sound_targets.json",
]

LOCAL_ARTIFACTS = [
    "models/source_kind_multilabel_v3_targeted.joblib",
    "models/synth_specialist_v4.joblib",
    "outputs/demucs_stems_6s_full/htdemucs_6s",
]

OPTIONAL_REPORTS = [
    "outputs/demucs_stems_6s_full/index.html",
    "outputs/demucs_stems_6s_full/broad_multilayer_timeline_multiscale.html",
    "outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.html",
]

IMPORT_MODULES = [
    "numpy",
    "soundfile",
    "librosa",
    "sklearn",
]


def exists(root: Path, rel: str) -> bool:
    return (root / rel).exists()


def check_paths(root: Path, title: str, paths: list[str], required: bool = True) -> int:
    print(f"\n[{title}]")
    missing = 0
    for rel in paths:
        ok = exists(root, rel)
        status = "ok" if ok else ("missing" if required else "not generated")
        print(f"{status:>13}  {rel}")
        if required and not ok:
            missing += 1
    return missing


def check_imports() -> int:
    print("\n[Python imports]")
    missing = 0
    for name in IMPORT_MODULES:
        ok = importlib.util.find_spec(name) is not None
        print(f"{'ok' if ok else 'missing':>13}  {name}")
        if not ok:
            missing += 1
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--strict-local-artifacts", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"project root: {root}")

    missing = 0
    missing += check_paths(root, "Required source files", REQUIRED_SOURCE_FILES, required=True)
    missing += check_paths(root, "Required config files", REQUIRED_CONFIG_FILES, required=True)
    missing += check_imports()
    local_missing = check_paths(root, "Local artifacts ignored by Git", LOCAL_ARTIFACTS, required=args.strict_local_artifacts)
    check_paths(root, "Optional generated reports", OPTIONAL_REPORTS, required=False)

    if local_missing and not args.strict_local_artifacts:
        print("\nlocal artifact note: missing local artifacts are expected on a fresh clone.")
        print("run with --strict-local-artifacts only on the original analysis machine.")

    if missing:
        raise SystemExit(f"\nsmoke check failed: {missing} required item(s) missing")
    print("\nsmoke check ok")


if __name__ == "__main__":
    main()
