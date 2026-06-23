# Clip Collection Plan

## Current Target

Build a profiler for dense alternative-pop / electronic-pop / hyperpop-adjacent clips where multiple layers are present at once. The model should not assume single-source audio. It should describe likely co-occurring layers and sound-design families.

## Current Bias

The current 34 clips appear biased toward:

- processed vocals and vocal-synth hybrids
- digital synth textures
- bell/pluck/game-like hooks
- glitch percussion and noisy FX
- 808/sub/sidechain-adjacent rhythm sections
- filtered or resampled melodic loops
- some guitar/plucked-string texture

This bias is acceptable for the first product-like model.

## Add Next

Prioritize clips that are close to the current sound world but expose missing distinctions.

### Highest Priority

1. Processed vocal clips
   - hard-tuned lead vocal
   - pitched vocal chop
   - formant-shifted vocal
   - vocoder or synthetic vocal
   - breathy background vocal pad

2. Synth-family clips
   - airy or washed synth pad
   - digital pluck / bell hook
   - bitcrushed lead
   - noisy wavetable texture
   - game-like synth melody
   - filtered sample/synth loop

3. Rhythm and low-end clips
   - obvious 808/sub sections
   - sidechain-pumped synth/bass
   - glitch percussion
   - trap hat/snare/clap sections
   - sparse kickless pulse sections

### Medium Priority

4. Guitar / sample-loop hybrids
   - filtered guitar loop
   - chorus guitar wash
   - resampled guitar/synth loop
   - plucked string-like hook

5. FX and noise textures
   - digital glitch fills
   - risers and impact tails
   - vinyl/tape/noise beds
   - granular smear textures

## Avoid For Now

Avoid adding clips that are too far from the current target unless there is a clear reason:

- pure shoegaze wall-of-guitars
- metal / punk / hard rock sections
- acoustic folk or singer-songwriter material
- jazz, classical, orchestral-heavy clips
- clean live-band recordings
- isolated drum loops without processed pop context

Shoegaze-influenced material is useful only when it overlaps with the current goal: processed vocals, chorus guitar wash, distorted texture, or electronic-pop production. Pure shoegaze should wait.

## Suggested Batch Size

For the next data round, add 20 to 40 clips:

- 8 to 12 processed vocal clips
- 8 to 12 synth-family clips
- 5 to 8 rhythm/808/glitch clips
- 4 to 6 guitar/sample-loop hybrid clips
- 3 to 5 FX/noise texture clips

Each clip should be 4 to 10 seconds long. It is okay if clips contain multiple layers; that is the real target.

## Naming Recommendation

Use names that encode rough source intent:

- `vocal_hardtune_01.wav`
- `vocal_chop_01.wav`
- `synth_pluck_01.wav`
- `synth_pad_01.wav`
- `808_sidechain_01.wav`
- `glitch_perc_01.wav`
- `guitar_wash_01.wav`
- `sample_loop_01.wav`

The names are not ground truth, but they help later review and debugging.
