from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Bitcrush, Chorus, Distortion, HighpassFilter, LowpassFilter, Pedalboard, Reverb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.synthesis import (
    SAMPLE_RATE,
    SourceSpec,
    apply_highpass_rise,
    apply_lowpass_sweep,
    apply_sidechain_pumping,
    db_to_amp,
    make_source,
    normalize,
    to_stereo,
    widen_mono,
)


SOURCE_FAMILIES = [
    "synth",
    "bass",
    "guitar_like",
    "vocal_like",
    "processed_vocal",
    "vocal_chop",
    "washed_guitar",
    "texture_noise",
    "ambient_pad",
    "ambient_texture",
    "noise_fx",
]
BASE_SOURCE_BY_PROFILE = {
    "processed_vocal": "vocal_like",
    "vocal_chop": "vocal_like",
    "washed_guitar": "guitar_like",
    "texture_noise": "synth",
    "ambient_pad": "synth",
    "ambient_texture": "synth",
}
REVERB_CLASSES = ["dry", "short_room", "long_hall", "washed_out"]
DISTORTION_CLASSES = ["none", "mild_saturation", "heavy_distortion", "crushed"]
FILTER_CLASSES = ["none", "lowpass_static", "highpass_static", "lowpass_opening", "highpass_rise"]
STEREO_CLASSES = ["mono", "medium", "wide"]
MOTION_CLASSES = ["static", "filter_opening", "filter_rise", "sidechain_pumping"]
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def apply_reverb(audio: np.ndarray, label: str) -> np.ndarray:
    if label == "dry":
        return audio
    if label == "short_room":
        board = Pedalboard([Reverb(room_size=0.25, damping=0.65, wet_level=0.18, dry_level=0.9)])
    elif label == "long_hall":
        board = Pedalboard([Reverb(room_size=0.72, damping=0.35, wet_level=0.32, dry_level=0.78)])
    elif label == "washed_out":
        board = Pedalboard([Reverb(room_size=0.92, damping=0.22, wet_level=0.55, dry_level=0.52)])
    else:
        return audio
    return normalize(board(audio.astype(np.float32), SAMPLE_RATE))


def apply_distortion(audio: np.ndarray, label: str) -> np.ndarray:
    if label == "none":
        return audio
    if label == "mild_saturation":
        board = Pedalboard([Distortion(drive_db=8.0)])
    elif label == "heavy_distortion":
        board = Pedalboard([Distortion(drive_db=22.0)])
    elif label == "crushed":
        board = Pedalboard([Distortion(drive_db=16.0), Bitcrush(bit_depth=7)])
    else:
        return audio
    return normalize(board(audio.astype(np.float32), SAMPLE_RATE))


def apply_filter(audio: np.ndarray, label: str) -> np.ndarray:
    if label == "none":
        return audio
    if label == "lowpass_static":
        board = Pedalboard([LowpassFilter(cutoff_frequency_hz=1_650.0)])
        return normalize(board(audio.astype(np.float32), SAMPLE_RATE))
    if label == "highpass_static":
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=650.0)])
        return normalize(board(audio.astype(np.float32), SAMPLE_RATE))
    if label == "lowpass_opening":
        return apply_lowpass_sweep(audio, start_hz=420.0, end_hz=7_500.0)
    if label == "highpass_rise":
        return apply_highpass_rise(audio, start_hz=80.0, end_hz=2_800.0)
    return audio


def apply_stereo(audio: np.ndarray, label: str) -> np.ndarray:
    if label == "mono":
        return to_stereo(audio)
    if label == "medium":
        return widen_mono(audio, amount=0.38)
    if label == "wide":
        return widen_mono(audio, amount=0.78)
    return to_stereo(audio)


def base_source_family(source_family: str) -> str:
    return BASE_SOURCE_BY_PROFILE.get(source_family, source_family)


def load_dry_source(source_root: Path, source_family: str, duration: float) -> tuple[np.ndarray, str]:
    import librosa

    folder = source_root / base_source_family(source_family)
    files = [path for path in folder.glob("*") if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
    if not files:
        raise FileNotFoundError(
            f"No dry source files found in {folder}. Add WAV/FLAC/OGG/AIFF files or use --source-mode generated."
        )

    path = random.choice(files)
    audio, sr = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    if sr != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE)

    target_len = int(SAMPLE_RATE * duration)
    if len(mono) >= target_len:
        max_start = len(mono) - target_len
        start = random.randint(0, max_start) if max_start > 0 else 0
        mono = mono[start : start + target_len]
    else:
        repeats = int(np.ceil(target_len / max(1, len(mono))))
        mono = np.tile(mono, repeats)[:target_len]

    # Short fade prevents clicks when a clipped segment starts mid-waveform.
    fade_len = min(int(0.025 * SAMPLE_RATE), len(mono) // 4)
    if fade_len > 1:
        fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        mono[:fade_len] *= fade
        mono[-fade_len:] *= fade[::-1]

    return normalize(mono), str(path.as_posix())


def parse_allowed_sources(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    sources = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(sources) - set(SOURCE_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown source family in --allowed-sources: {unknown}")
    return sources


def discover_dry_sources(source_root: Path, allowed_sources: list[str] | None) -> list[str]:
    candidates = allowed_sources or SOURCE_FAMILIES
    available = []
    for source in candidates:
        folder = source_root / base_source_family(source)
        files = [path for path in folder.glob("*") if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS]
        if files:
            available.append(source)

    if not available:
        raise FileNotFoundError(
            f"No dry source files found under {source_root}. "
            "Add WAV/FLAC/OGG/AIFF files or use --source-mode generated."
        )
    return available


def choose_motion(filter_label: str, sidechain_prob: float) -> str:
    if filter_label == "lowpass_opening":
        return "filter_opening"
    if filter_label == "highpass_rise":
        return "filter_rise"
    if random.random() < sidechain_prob:
        return "sidechain_pumping"
    return "static"


def choose_labels(effect_stage: int, sampling: str) -> tuple[str, str, str, str]:
    if sampling == "balanced":
        reverb = random.choice(REVERB_CLASSES)
        distortion = random.choice(DISTORTION_CLASSES)
        filter_label = random.choice(FILTER_CLASSES)
        stereo = random.choice(STEREO_CLASSES)
        if effect_stage <= 1:
            # Keep stage 1 from becoming too dense while still giving every
            # classifier head non-trivial examples.
            active = [reverb != "dry", distortion != "none", filter_label != "none"]
            if sum(active) > 1:
                keep = random.choice([idx for idx, value in enumerate(active) if value])
                if keep != 0:
                    reverb = "dry"
                if keep != 1:
                    distortion = "none"
                if keep != 2:
                    filter_label = "none"
        return reverb, distortion, filter_label, stereo

    reverb = random.choice(REVERB_CLASSES)
    distortion = random.choice(DISTORTION_CLASSES)
    filter_label = random.choice(FILTER_CLASSES)
    stereo = random.choice(STEREO_CLASSES)

    if effect_stage <= 1:
        active_groups = random.sample(["reverb", "distortion", "filter"], k=random.randint(0, 1))
    else:
        active_groups = random.sample(["reverb", "distortion", "filter"], k=random.randint(0, 2))

    if "reverb" not in active_groups:
        reverb = "dry"
    if "distortion" not in active_groups:
        distortion = "none"
    if "filter" not in active_groups:
        filter_label = "none"
    return reverb, distortion, filter_label, stereo


def adapt_labels_for_profile_source(
    source: str,
    reverb: str,
    distortion: str,
    filter_label: str,
    stereo: str,
) -> tuple[str, str, str, str, str | None]:
    modulation: str | None = None

    if source == "processed_vocal":
        if distortion == "none":
            distortion = random.choice(["mild_saturation", "heavy_distortion", "crushed"])
        if reverb == "dry" and random.random() < 0.55:
            reverb = random.choice(["short_room", "long_hall"])
        stereo = random.choice(["medium", "wide"])
        modulation = "chorus" if random.random() < 0.55 else None

    elif source == "vocal_chop":
        if distortion == "none":
            distortion = random.choice(["mild_saturation", "crushed"])
        if filter_label == "none" and random.random() < 0.55:
            filter_label = random.choice(["highpass_static", "highpass_rise", "lowpass_static"])
        reverb = random.choice(["dry", "short_room", "long_hall"])
        stereo = random.choice(["medium", "wide"])
        modulation = "chorus" if random.random() < 0.35 else None

    elif source == "washed_guitar":
        reverb = random.choice(["long_hall", "washed_out"])
        if filter_label == "none" and random.random() < 0.70:
            filter_label = random.choice(["lowpass_static", "lowpass_opening"])
        stereo = "wide"
        modulation = "chorus"

    elif source == "texture_noise":
        reverb = random.choice(["long_hall", "washed_out"])
        distortion = random.choice(["mild_saturation", "crushed", "none"])
        if filter_label == "none" and random.random() < 0.75:
            filter_label = random.choice(["highpass_static", "highpass_rise", "lowpass_static"])
        stereo = "wide"
        modulation = "chorus" if random.random() < 0.30 else None

    elif source == "ambient_pad":
        reverb = random.choice(["long_hall", "washed_out"])
        if filter_label == "none" and random.random() < 0.70:
            filter_label = random.choice(["lowpass_static", "lowpass_opening"])
        if distortion != "none" and random.random() < 0.70:
            distortion = "none"
        stereo = "wide"
        modulation = "chorus" if random.random() < 0.60 else None

    elif source == "ambient_texture":
        reverb = random.choice(["long_hall", "washed_out"])
        if filter_label == "none" and random.random() < 0.65:
            filter_label = random.choice(["lowpass_static", "lowpass_opening", "highpass_rise"])
        if distortion != "none" and random.random() < 0.55:
            distortion = "none"
        stereo = "wide"
        modulation = "chorus" if random.random() < 0.45 else None

    return reverb, distortion, filter_label, stereo, modulation


def make_base_audio(source: str, role: str, melody: str, articulation: str, duration: float, source_mode: str, dry_source_root: Path) -> tuple[np.ndarray, str, str | None]:
    if source_mode == "dry":
        audio, source_file = load_dry_source(dry_source_root, source, duration)
        return audio, "dry_file", source_file

    spec = SourceSpec(source=base_source_family(source), role=role, melody=melody, articulation=articulation)
    return make_source(spec, duration), "generated", None


def render_loop(
    index: int,
    out_dir: Path,
    duration: float,
    effect_stage: int,
    sampling: str,
    sidechain_prob: float,
    source_mode: str,
    dry_source_root: Path,
    source_choices: list[str],
) -> dict:
    source = random.choice(source_choices)
    role = "bass_foundation" if source == "bass" else random.choice(["main_melody", "background_texture"])
    articulation = random.choice(["sustained", "plucked", "chopped", "pulsing"])
    melody = "non_melodic" if source == "noise_fx" else random.choice(["melody_active", "motif_repeated", "non_melodic"])
    if source == "vocal_chop":
        role = random.choice(["main_melody", "background_texture"])
        articulation = "chopped"
        melody = random.choice(["motif_repeated", "melody_active"])
    elif source == "texture_noise":
        role = "background_texture"
        articulation = random.choice(["swelling", "pulsing", "sustained"])
        melody = "non_melodic"
    elif source == "ambient_pad":
        role = "background_texture"
        articulation = random.choice(["sustained", "swelling", "pulsing"])
        melody = random.choice(["non_melodic", "motif_repeated"])

    reverb, distortion, filter_label, stereo = choose_labels(effect_stage, sampling)
    reverb, distortion, filter_label, stereo, forced_modulation = adapt_labels_for_profile_source(
        source, reverb, distortion, filter_label, stereo
    )

    motion = choose_motion(filter_label, sidechain_prob)

    audio, source_origin, source_file = make_base_audio(source, role, melody, articulation, duration, source_mode, dry_source_root)
    audio = apply_distortion(audio, distortion)
    audio = apply_filter(audio, filter_label)
    audio = apply_reverb(audio, reverb)

    if motion == "sidechain_pumping":
        audio = apply_sidechain_pumping(audio, bpm=120.0)

    if forced_modulation == "chorus" or (forced_modulation is None and random.random() < 0.22):
        audio = normalize(Pedalboard([Chorus(rate_hz=1.2, depth=0.55, mix=0.35)])(audio.astype(np.float32), SAMPLE_RATE))
        modulation = "chorus"
    else:
        modulation = "none"

    audio = apply_stereo(audio, stereo)
    gain_db = random.uniform(-3.0, -0.5)
    audio = normalize(audio * db_to_amp(gain_db), peak=0.92)

    filename = f"loop_{index:06d}.wav"
    sf.write(out_dir / "audio" / filename, audio.T, SAMPLE_RATE)

    return {
        "file": f"audio/{filename}",
        "duration": duration,
        "source_family": source,
        "source_origin": source_origin,
        "source_file": source_file,
        "role": role,
        "articulation": articulation,
        "melody": melody,
        "reverb": reverb,
        "distortion": distortion,
        "filter": filter_label,
        "filter_presence": "none" if filter_label == "none" else "filtered",
        "filter_motion_type": "dynamic" if filter_label in {"lowpass_opening", "highpass_rise"} else ("static" if filter_label in {"lowpass_static", "highpass_static"} else "none"),
        "stereo": stereo,
        "motion": motion,
        "motion_presence": "motion" if motion != "static" else "static",
        "modulation": modulation,
        "effect_stage": effect_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--effect-stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--sampling", choices=["balanced", "curriculum"], default="balanced")
    parser.add_argument("--sidechain-prob", type=float, default=0.35)
    parser.add_argument("--source-mode", choices=["generated", "dry"], default="generated")
    parser.add_argument("--dry-source-root", type=Path, default=Path("data/dry_sources"))
    parser.add_argument("--allowed-sources", default=None, help="Comma-separated source families, e.g. bass,guitar_like,synth,vocal_like")
    parser.add_argument("--out", type=Path, default=Path("data/loop_synthetic"))
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    allowed_sources = parse_allowed_sources(args.allowed_sources)
    source_choices = discover_dry_sources(args.dry_source_root, allowed_sources) if args.source_mode == "dry" else (allowed_sources or SOURCE_FAMILIES)
    print(f"source choices: {', '.join(source_choices)}")

    metadata_path = args.out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for index in range(args.count):
            row = render_loop(
                index,
                args.out,
                args.duration,
                args.effect_stage,
                args.sampling,
                args.sidechain_prob,
                args.source_mode,
                args.dry_source_root,
                source_choices,
            )
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"wrote {args.count} loop samples")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
