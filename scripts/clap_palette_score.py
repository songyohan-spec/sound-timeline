from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: Path, target_sr: int = 48_000, min_seconds: float = 10.0) -> tuple[np.ndarray, int]:
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
    min_len = int(min_seconds * sr)
    if len(mono) < min_len:
        mono = np.pad(mono, (0, min_len - len(mono)))
    return mono, sr


def flatten_palette(palette: dict) -> tuple[list[str], list[tuple[str, str, str]]]:
    texts: list[str] = []
    keys: list[tuple[str, str, str]] = []
    for family, labels in palette.items():
        for label, prompts in labels.items():
            for prompt in prompts:
                texts.append(prompt)
                keys.append((family, label, prompt))
    return texts, keys


def load_clap(model_name: str):
    from transformers import ClapModel, ClapProcessor

    try:
        processor = ClapProcessor.from_pretrained(model_name, local_files_only=True)
        model = ClapModel.from_pretrained(model_name, local_files_only=True)
    except Exception:
        processor = ClapProcessor.from_pretrained(model_name)
        model = ClapModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def feature_tensor(output):
    if hasattr(output, "pooler_output"):
        return output.pooler_output
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "audio_embeds"):
        return output.audio_embeds
    return output


def score_palette(audio_path: Path, palette_path: Path, model_name: str) -> dict:
    return PaletteScorer(palette_path, model_name).score(audio_path)


class PaletteScorer:
    def __init__(self, palette_path: Path, model_name: str) -> None:
        palette = json.loads(palette_path.read_text(encoding="utf-8"))
        self.texts, self.keys = flatten_palette(palette)
        self.model_name = model_name
        self.processor, self.model = load_clap(model_name)

    def score(self, audio_path: Path) -> dict:
        audio, sr = load_audio(audio_path)
        return score_audio_with_loaded_model(
            audio_path,
            audio,
            sr,
            self.texts,
            self.keys,
            self.processor,
            self.model,
            self.model_name,
        )


def score_audio_with_loaded_model(
    audio_path: Path,
    audio: np.ndarray,
    sr: int,
    texts: list[str],
    keys: list[tuple[str, str, str]],
    processor,
    model,
    model_name: str,
) -> dict:
    import torch

    inputs = processor(text=texts, audio=[audio], sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_audio[0]
        probs = logits.softmax(dim=0).cpu().numpy()

    return build_palette_report(audio_path, keys, probs, model_name)


def build_palette_report(audio_path: Path, keys: list[tuple[str, str, str]], scores: np.ndarray, model_name: str) -> dict:
    grouped: dict[str, dict[str, dict]] = {}
    for (family, label, prompt), score in zip(keys, scores):
        entry = grouped.setdefault(family, {}).setdefault(label, {"label": label, "score": 0.0, "prompt": ""})
        if float(score) > entry["score"]:
            entry["score"] = float(score)
            entry["prompt"] = prompt

    families = {}
    all_items = []
    for family, labels in grouped.items():
        ranked = sorted(labels.values(), key=lambda item: item["score"], reverse=True)
        for item in ranked:
            item["score"] = round(float(item["score"]), 8)
            all_items.append({"family": family, **item})
        families[family] = ranked

    all_items = sorted(all_items, key=lambda item: item["score"], reverse=True)
    return {
        "audio": str(audio_path),
        "model": model_name,
        "families": families,
        "top_overall": all_items[:20],
        "caution": "CLAP palette scores are prompt rankings, not calibrated probabilities or ground truth.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--palette", type=Path, default=Path("configs/sound_palette_prompts.json"))
    parser.add_argument("--model-name", default="laion/clap-htsat-unfused")
    parser.add_argument("--out", type=Path, default=Path("outputs/palette_scores.json"))
    args = parser.parse_args()

    report = score_palette(args.audio, args.palette, args.model_name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
