from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int = 48_000) -> tuple[np.ndarray, int]:
    import librosa

    try:
        audio, sr = sf.read(path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
    except Exception:
        mono, sr = librosa.load(path, sr=None, mono=True)
        mono = mono.astype(np.float32)

    if sr != target_sr:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return mono, sr


def write_segment(audio: np.ndarray, sr: int, start: float, duration: float, out_path: Path) -> None:
    start_sample = int(start * sr)
    end_sample = start_sample + int(duration * sr)
    clip = audio[start_sample:min(end_sample, len(audio))]
    if len(clip) < int(duration * sr):
        clip = np.pad(clip, (0, int(duration * sr) - len(clip)))
    sf.write(out_path, clip, sr)


def flatten_prompts(prompt_config: dict) -> tuple[list[str], list[tuple[str, str]]]:
    texts: list[str] = []
    keys: list[tuple[str, str]] = []
    for group, labels in prompt_config.items():
        for label, prompts in labels.items():
            for prompt in prompts:
                texts.append(prompt)
                keys.append((group, label))
    return texts, keys


def score_with_transformers(audio_path: Path, prompt_config: dict, model_name: str) -> dict:
    import torch
    from transformers import ClapModel, ClapProcessor

    audio, sr = load_audio(audio_path, target_sr=48_000)
    texts, keys = flatten_prompts(prompt_config)

    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)
    model.eval()

    inputs = processor(text=texts, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_audio[0]
        probs = logits.softmax(dim=0).cpu().numpy()

    grouped: dict[str, dict[str, list[float]]] = {}
    for (group, label), score in zip(keys, probs):
        grouped.setdefault(group, {}).setdefault(label, []).append(float(score))

    result = {}
    for group, labels in grouped.items():
        ranked = []
        for label, scores in labels.items():
            ranked.append({"label": label, "score": round(float(max(scores)), 6)})
        result[group] = sorted(ranked, key=lambda item: item["score"], reverse=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=Path("configs/clap_prompts.json"))
    parser.add_argument("--model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--out", type=Path, default=Path("outputs/clap_scores.json"))
    args = parser.parse_args()

    if not args.audio.exists():
        raise SystemExit(f"Audio not found: {args.audio}")
    prompt_config = json.loads(args.prompts.read_text(encoding="utf-8"))

    try:
        scores = score_with_transformers(args.audio, prompt_config, args.model_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency for CLAP scoring. Install it with:\n"
            "pip install transformers accelerate\n"
            "Then rerun this script."
        ) from exc

    report = {
        "audio": str(args.audio),
        "model": args.model_name,
        "scores": scores,
        "caution": "CLAP scores are semantic similarity hints, not ground truth labels.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
