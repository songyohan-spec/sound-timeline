# Local Artifacts

The GitHub repository intentionally does not include generated data, trained
models, Demucs stems, or audio outputs.

## Required Local Folders

These folders are ignored by Git:

- `data/`
- `models/`
- `outputs/`

## Expected Important Files

The current best pipeline expects these local artifacts:

### Models

- `models/source_kind_multilabel_v3_targeted.joblib`
- `models/synth_specialist_v4.joblib`

Optional but useful:

- `models/source_kind_multilabel_v2.joblib`
- `models/reference_mixture_multilabel_v5_vocal.joblib`
- `models/macro_reranker_v3.joblib`

### Demucs Stem Outputs

- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/vocals.wav`
- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/drums.wav`
- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/bass.wav`
- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/guitar.wav`
- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/piano.wav`
- `outputs/demucs_stems_6s_full/htdemucs_6s/<track>/other.wav`

### Main Generated Reports

- `outputs/demucs_stems_6s_full/index.html`
- `outputs/demucs_stems_6s_full/broad_multilayer_timeline_multiscale.html`
- `outputs/demucs_stems_6s_full/broad_layer_regions_likely_multiscale.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_strict_index.html`
- `outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.html`

## Why These Are Not Committed

Many files are large, generated, or derived from copyrighted audio. They should
remain local unless intentionally packaged as a private artifact bundle.
