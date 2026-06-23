# Day 2 Summary

## Current Scope

The project is currently a loop-level sound design profiling MVP. It does not
perform source separation or exact VST parameter recovery yet.

## What Works

- NSynth dry-source import into `data/dry_sources`
- Dry-source loop dataset generation
- RandomForest baseline with DSP/stereo/band-motion features
- JSON inference
- Markdown/HTML loop report rendering
- Segment-level inference for external clips

## Strong Baseline Targets

The coarse RandomForest model is currently the strongest path:

- `source_family`
- `reverb`
- `distortion`
- `filter_presence`
- `filter_motion_type`
- `stereo`
- `motion_presence`

## Known Weak Spots

- Fine-grained lowpass opening remains harder than coarse dynamic/static filter motion.
- Mild saturation remains ambiguous against naturally colored dry sources.
- The CNN baseline underperforms the DSP-feature RandomForest on the current data size.
- Real alternative music clips are still expected to expose domain gap.

## Next Experiment

Use real 3-8 second external clips and inspect which labels fail first.

