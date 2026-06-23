# Current Pipeline Status

## Goal

Analyze short mixed music clips as layered sound-design material, not as single-source audio.

Current project framing:

1. Separate each clip into Demucs stems. Default: `vocals`, `drums`, `bass`, `other`; optional 6-stem mode: `vocals`, `drums`, `bass`, `guitar`, `piano`, `other`.
2. Analyze each stem in 2-second windows.
3. Output stem-aware source-kind candidates such as vocal chop, processed vocal, electronic drum machine, 808/sub bass, synth bass, sampled loop texture, filtered/muffled loop, glitch percussion, guitar/keys/synth texture.
4. Treat results as sound-layer hypotheses, not exact plugin, preset, or source-separation truth.

## Recommended Command

```powershell
C:\Users\Song\miniconda3\envs\sound-timeline\python.exe scripts\run_demucs_source_kind_pipeline.py --out-root outputs\demucs_stems_full --skip-separation --model models\source_kind_multilabel_v2.joblib --threshold-scale 0.85 --quality librosa
```

Use without `--skip-separation` when new files are added and Demucs stems have not been generated yet:

```powershell
C:\Users\Song\miniconda3\envs\sound-timeline\python.exe scripts\run_demucs_source_kind_pipeline.py --input-dir data\external_loops --out-root outputs\demucs_stems_full --model models\source_kind_multilabel_v2.joblib --threshold-scale 0.85 --quality librosa
```

## More Detailed Stem Split

The default Demucs model `htdemucs` produces `vocals`, `drums`, `bass`, and `other`.

For more separation detail, `htdemucs_6s` can produce `vocals`, `drums`, `bass`, `guitar`, `piano`, and `other`.

Probe command:

```powershell
C:\Users\Song\miniconda3\envs\sound-timeline\python.exe scripts\run_demucs_source_kind_pipeline.py --input-dir data\external_loops --out-root outputs\demucs_stems_6s --demucs-model htdemucs_6s --model models\source_kind_multilabel_v2.joblib --threshold-scale 0.85 --quality librosa
```

`htdemucs_6s` has been run over the full current 83-clip collection and produced `guitar.wav` and `piano.wav` stems.

## Active Segment Meaning

In `collection_summary.html`, `active` means a 2-second stem segment produced at least one stem-valid source-kind candidate above the current threshold.

Inactive does not always mean silence. It can mean:

- Demucs placed little useful signal into that stem.
- The model saw possible candidates, but all were below threshold.
- Stem-aware gating suppressed candidates that did not belong to that stem.
- The segment contains a sound type that the current source-kind model does not cover well.

## Main Outputs

- `outputs/demucs_stems_full/index.html`
- `outputs/demucs_stems_full/stem_timeline_matrix.html`
- `outputs/demucs_stems_full/stem_source_kind.html`
- `outputs/demucs_stems_full/source_kind_gap_audit.html`
- `outputs/demucs_stems_6s_full/index.html`
- `outputs/demucs_stems_6s_probe/index.html`

## Current Model

Recommended model:

- `models/source_kind_multilabel_v2.joblib`

Training features:

- Base source-kind features: `outputs/source_kind_features_v1.csv`
- Weak Demucs stem examples: `outputs/weak_source_kind_features.csv`
- Combined v2 features: `outputs/source_kind_features_v2_weak.csv`

The v2 model uses weak examples harvested from the user's own Demucs stems for previously under-supported source kinds:

- `electronic_drum_machine`
- `glitch_percussion`
- `sampled_loop_texture`
- `piano_or_keyboard_loop`
- `warm_keys_or_organ`

## Current Coverage

After weak harvesting and v2 rerun:

- `ok`: 29 source kinds
- `not_seen`: 12 source kinds
- `weak_detector_for_stem`: 1 source kind

Remaining weak detector:

- `bitcrushed_or_aliasing_synth`

## Important Limitations

This is not a true ground-truth source separator or exact sound-design reverse-engineer.

Known risks:

- Demucs can place vocal chops or processed effects into `other`, not always `vocals`.
- Bass stem may contain kick-like low hits.
- Drum stem detail is still coarse compared with real drum taxonomy.
- Weak examples improve domain fit but can reinforce model mistakes if harvested blindly.
- AST/AudioSet is useful conceptually but too slow in the current CPU Windows setup to be a core loop.

## Next Completeness Tasks

Prioritize system completeness over tiny labels:

1. Compare `outputs/demucs_stems_full` against `outputs/demucs_stems_6s_full` and decide whether 6-stem should become the default.
2. Add a "new clips" workflow: drop files into `data/external_loops`, run one command, get refreshed reports.
3. Add a small manual override CSV only for obvious model mistakes, not full annotation.
4. Later, add stronger open-source panels only if they run fast enough locally or can be cached cleanly.
