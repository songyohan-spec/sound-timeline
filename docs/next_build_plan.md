# Next Build Plan

This document defines the next engineering steps after the initial GitHub
upload. The goal is to turn the current exploratory pipeline into a reproducible
project skeleton before adding more labels or heavier models.

## Current Honest Status

The project can:

- Split short music clips into Demucs stems.
- Analyze stems with source-kind and synth-specialist models.
- Compare 2-second synth hints against 4-second context.
- Render broad layer timelines and region summaries.
- Export pseudo-separated synth candidates for audition.
- Triage strict synth candidates into auditionable / too broad / silent /
  empty buckets.

The project cannot yet:

- Produce ground-truth synth stems from a mixed commercial song.
- Reliably distinguish every synth subtype such as granular vs wavetable vs FM
  without false positives.
- Infer exact synth plugin, preset, oscillator, or effect-chain parameters.
- Replace human listening for final correctness.

## Build Priorities

### P0: Make The Current Pipeline Reproducible

Required:

- Keep `scripts/run_current_best_synth_pipeline.py` as the main command.
- Keep `outputs/`, `models/`, and `data/` out of Git.
- Maintain a smoke test that checks required source files, configs, and docs.
- Document the required local artifacts separately from committed code.

Done when:

- A fresh clone shows exactly which files are missing because they are local
  artifacts.
- Running the smoke test clearly says what must be generated or downloaded.

### P1: Separate "Reports" From "Model Core"

Current scripts mix three concerns:

- Feature extraction
- Model inference / filtering
- HTML report rendering

Next refactor target:

- Move reusable feature helpers into `src/sound_timeline/features.py`.
- Move audio I/O and segment helpers into a new `src/sound_timeline/audio.py`.
- Keep scripts as thin command-line wrappers.

Done when:

- New scripts mostly parse args and call package functions.
- The same logic is not copied across multiple scripts.

### P2: Make Synth Candidate Export More Useful

Current pseudo-separation is stem-region masking, not true source separation.
The immediate improvement is not a bigger label list. It is better candidate
selection.

Next work:

- Use triage results to suppress `silent_selection` and `empty_candidate`.
- Add a stricter export mode that keeps only `auditionable` candidates.
- Compare `synth_candidate.wav` against `residual_context.wav` in a report.
- Add per-track notes only for candidate failures, not every segment.

Done when:

- The strict audition page mostly contains clips worth listening to.
- Bad candidates are not silently mixed into the main result.

Current immediate output:

- `outputs/demucs_stems_6s_full/synth_candidate_audio_auditionable.html`

### P3: Improve Model Evidence

Only after P0-P2:

- Add more weak examples for under-supported synth subtypes.
- Use cached open-source teacher signals only if they are fast enough.
- Prefer small, inspectable teacher queues over blind auto-labeling.

High-risk labels:

- `granular_texture`
- `wavetable_noise`
- `fuzzy_lofi_synth`
- `formant_vocoder`
- `bitcrushed_synth_lead`

These labels are valuable, but easy to hallucinate from noisy separated stems.

## Recommended Next Command

After adding or changing clips:

```powershell
python scripts\run_current_best_synth_pipeline.py
python scripts\smoke_check_project.py
```

Then open:

- `outputs/demucs_stems_6s_full/index.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.html`

## What Not To Do Next

Do not expand the label ontology just because a broad label feels unsatisfying.
Every new label needs either:

- credible synthetic examples,
- credible stem examples,
- or a cached external teacher signal.

Otherwise the model will look more detailed while becoming less truthful.
