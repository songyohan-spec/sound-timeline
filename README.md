# Sound Timeline

Timeline-based sound layer and production-effect profiling for modern music.

## Current MVP

The first milestone is not full song analysis yet. It is a synthetic data
generator that creates short audio clips with controlled labels:

- source family: synth, bass, guitar-like, vocal-like, noise_fx
- role: main_melody, background_texture, bass_foundation, transition_fx
- processing: reverb, delay, distortion, lowpass_filter, bitcrush, chorus
- texture/spatial: dry_close, wide, washed_out, grainy, pumped

These clips become the ground-truth training data for the first baseline model.

## Environment

Activate your local Conda environment:

```powershell
conda activate sound-timeline
cd C:\Users\Song\Documents\Codex\2026-04-29\ai-x
```

Verify dependencies:

```powershell
python -c "import numpy, pandas, scipy, matplotlib, librosa, soundfile, pedalboard; print('audio env ok')"
```

## Generate Synthetic Data

```powershell
python scripts\generate_synthetic_dataset.py --count 64 --out data\synthetic
```

This writes WAV files and a `metadata.jsonl` label file under `data\synthetic`.

## Inspect Generated Data

Render spectrogram previews and a small local HTML listening page:

```powershell
python scripts\inspect_dataset.py --dataset data\synthetic --count 12
```

Open `data\synthetic\preview.html` in your browser and check whether the labels
match what you hear.

## Train a Tiny Baseline

This first baseline uses hand-crafted audio features and logistic regression.
It is a sanity check, not the final model.

```powershell
python scripts\generate_synthetic_dataset.py --count 256 --out data\synthetic
python scripts\train_baseline.py --dataset data\synthetic
```

## Generate Timeline Synthetic Data

This creates short mix-level examples with multiple overlapping regions,
dynamic effects, gain/prominence labels, and region-level metadata.

```powershell
python scripts\generate_timeline_dataset.py --count 16 --out data\timeline_synthetic
```

Preview the generated mix-level timeline labels:

```powershell
python scripts\inspect_timeline_dataset.py --dataset data\timeline_synthetic --count 8
```

Open `data\timeline_synthetic\timeline_preview.html` in your browser.

## Loop-Level MVP

The current main direction is loop-level sound design profiling. Generate a
controlled loop dataset:

```powershell
python scripts\generate_loop_dataset.py --count 256 --effect-stage 1 --sampling balanced --out data\loop_synthetic
python scripts\summarize_loop_dataset.py --dataset data\loop_synthetic
python scripts\inspect_loop_dataset.py --dataset data\loop_synthetic --count 12
python scripts\train_loop_baseline.py --dataset data\loop_synthetic
```

To use real dry sources, first create the folder layout and place audio files
inside the matching source-family folders:

```powershell
python scripts\prepare_dry_source_dirs.py --root data\dry_sources
```

Then generate from those files:

```powershell
python scripts\generate_loop_dataset.py --count 1000 --source-mode dry --dry-source-root data\dry_sources --effect-stage 1 --sampling balanced --sidechain-prob 0.45 --out data\loop_dry_stage1
```

You can also bootstrap dry sources from the NSynth test split:

```powershell
python scripts\import_nsynth_dry_sources.py --download --max-per-family 50 --out-root data\dry_sources
python scripts\summarize_dry_sources.py --root data\dry_sources
python scripts\generate_loop_dataset.py --count 2000 --source-mode dry --dry-source-root data\dry_sources --effect-stage 1 --sampling balanced --sidechain-prob 0.45 --out data\loop_nsynth_stage1
python scripts\train_loop_baseline.py --dataset data\loop_nsynth_stage1
```

Run inference on a loop:

```powershell
python scripts\infer_loop.py --model models\loop_baseline.joblib --audio data\loop_nsynth_stage1\audio\loop_000000.wav
```

Render the JSON inference result into a readable report:

```powershell
python scripts\infer_loop.py --model models\loop_baseline.joblib --audio data\loop_nsynth_stage1\audio\loop_000000.wav --out outputs\loop_000000_report.json
python scripts\render_loop_report.py --input outputs\loop_000000_report.json --out-md outputs\loop_000000_report.md --out-html outputs\loop_000000_report.html
```

Analyze a longer audio file as fixed-length segments:

```powershell
python scripts\infer_audio_segments.py --model models\loop_baseline.joblib --audio path\to\your_loop_or_song.wav --segment-seconds 4 --hop-seconds 4 --out outputs\segments_report.json
python scripts\render_segments_report.py --input outputs\segments_report.json --out-html outputs\segments_report.html
python scripts\pseudo_label_segments.py --input outputs\segments_report.json --out outputs\segments_pseudo_labels.csv
python scripts\render_pseudo_labels.py --input outputs\segments_pseudo_labels.csv --out-html outputs\segments_pseudo_labels.html
```

Cut a short clip from a longer file, then analyze it:

```powershell
python scripts\cut_audio_clip.py --input path\to\song.mp3 --start 42 --duration 8 --out data\external_loops\song_042_050.wav
python scripts\analyze_external_clip.py --audio data\external_loops\song_042_050.wav --name song_042_050
```

Batch analyze every audio file in `data\external_loops`:

```powershell
python scripts\batch_analyze_external.py --input-dir data\external_loops --out-dir outputs\external_batch
python scripts\summarize_external_batch.py --batch-dir outputs\external_batch
python scripts\render_case_study_report.py --batch-dir outputs\external_batch --title "External Clip Set"
```

Optional CLAP semantic scoring panel:

```powershell
pip install transformers accelerate
python scripts\clap_score_audio.py --audio data\external_loops\your_clip.wav --out outputs\your_clip_clap.json
python scripts\render_clap_report.py --input outputs\your_clip_clap.json --out-html outputs\your_clip_clap.html
python scripts\render_ensemble_report.py --segments outputs\segments_report.json --clap outputs\your_clip_clap.json --title "External Clip"
python scripts\batch_clap_external.py --input-dir data\external_loops --out-dir outputs\external_clap
python scripts\render_batch_ensemble_summary.py --rf-summary outputs\external_batch_alt2\summary.csv --clap-dir outputs\external_clap
```

Higher-resolution open-vocabulary sound palette retrieval:

```powershell
python scripts\clap_palette_score.py --audio data\external_loops\your_clip.wav --out outputs\your_clip_palette.json
python scripts\render_palette_report.py --input outputs\your_clip_palette.json --out-html outputs\your_clip_palette.html
python scripts\render_sound_profile_report.py --palette outputs\your_clip_palette.json --segments outputs\segments_report.json --title "External Clip"
```

One-command sound profile JSON pipeline:

```powershell
python scripts\analyze_sound_profile.py --audio data\external_loops\your_clip.wav --name your_clip
python scripts\batch_analyze_sound_profiles.py --input-dir data\external_loops --out-dir outputs\sound_profiles
```

Open-vocabulary sound palette timeline for a short clip:

```powershell
python scripts\analyze_sound_palette_timeline.py --audio data\external_loops\your_clip.wav --segment-seconds 2 --hop-seconds 2
python scripts\batch_analyze_sound_palette_timelines.py --input-dir data\external_loops --out-dir outputs\palette_timelines --segment-seconds 2 --hop-seconds 2
python scripts\batch_calibrate_palette_timelines.py --timeline-dir outputs\palette_timelines --out-dir outputs\calibrated_timelines
python scripts\export_macro_reranker_dataset.py --palette-dir outputs\palette_timelines --calibrated-dir outputs\calibrated_timelines --out outputs\macro_reranker_dataset.csv
python scripts\train_macro_reranker.py --dataset outputs\macro_reranker_dataset.csv --out models\macro_reranker.joblib
python scripts\batch_apply_macro_reranker.py --timeline-dir outputs\palette_timelines --model models\macro_reranker.joblib --out-dir outputs\reranked_timelines
```

Train the first mel-spectrogram CNN baseline:

```powershell
python scripts\train_loop_cnn.py --dataset data\loop_nsynth_stage1 --epochs 8 --batch-size 32
```

## Current Demucs Stem + Synth Candidate Pipeline

The current practical pipeline is no longer a single-loop classifier. It is a
stem-aware analysis flow for short external music clips:

1. Split clips with Demucs `htdemucs_6s`.
2. Analyze each stem in 2-second windows.
3. Cross-check synth-like labels with a 4-second context window.
4. Render broad layer timelines and region summaries.
5. Export auditionable pseudo-separated synth candidates.

Run the current best pipeline after Demucs stems have been generated:

```powershell
python scripts\run_current_best_synth_pipeline.py
```

Check the project skeleton:

```powershell
python scripts\smoke_check_project.py
```

Main generated reports:

- `outputs/demucs_stems_6s_full/index.html`
- `outputs/demucs_stems_6s_full/broad_multilayer_timeline_multiscale.html`
- `outputs/demucs_stems_6s_full/broad_layer_regions_likely_multiscale.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_index.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_strict_index.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_auditionable.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_failure_audit.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_reliable.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_final.html`
- `outputs/demucs_stems_6s_full/current_best_status.html`

The synth candidate audio exports are pseudo-separation outputs. They are made
by combining Demucs stems over model-selected time regions. They are useful for
auditioning whether the model is pulling the right synth-like family, but they
are not ground-truth isolated synthesizer stems.

Start with `synth_candidate_audio_final.html` for listening, then use
`current_best_status.html` to see how many candidates were rejected or retained
and which synth labels dominate the current build.

## Repository / GitHub Notes

The repository is set up so generated data, trained models, Demucs stems, audio
exports, and HTML reports are ignored by Git:

- `data/`
- `models/`
- `outputs/`
- audio files such as `*.wav`, `*.mp3`, `*.flac`

This keeps GitHub uploads lightweight and avoids accidentally committing large
or copyrighted audio. To share a reproducible snapshot, commit the source code,
configs, docs, `README.md`, `environment.yml`, and `.gitignore`; keep generated
artifacts local or publish them separately as a release/archive only when needed.

If `git` is not available in the current terminal, install Git for Windows or
open a terminal where `git --version` works before running:

```powershell
git status
git add .gitignore README.md environment.yml scripts configs docs src
git commit -m "Add stem-aware synth profiling pipeline"
```

See also:

- `docs/next_build_plan.md`
- `docs/local_artifacts.md`
- `docs/github_upload_checklist.md`
