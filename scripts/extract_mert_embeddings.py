from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def discover_audio(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)


def load_audio(path: Path, target_sr: int) -> np.ndarray:
    try:
        audio, sr = sf.read(path, always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
    except Exception:
        import librosa

        mono, sr = librosa.load(path, sr=None, mono=True)
        mono = mono.astype(np.float32)
    if sr != target_sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    max_abs = float(np.max(np.abs(mono))) if len(mono) else 0.0
    if max_abs > 1e-6:
        mono = mono / max_abs
    return mono


def load_mert(model_name: str, device: str):
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    model.to(device)
    return processor, model, torch


def embed_audio(path: Path, processor, model, torch, sample_rate: int, device: str, max_seconds: float) -> np.ndarray:
    audio = load_audio(path, sample_rate)
    max_len = int(sample_rate * max_seconds)
    if len(audio) > max_len:
        start = max(0, (len(audio) - max_len) // 2)
        audio = audio[start : start + max_len]
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model(**inputs, output_hidden_states=True)
    hidden = output.last_hidden_state
    pooled = hidden.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = float(np.linalg.norm(pooled))
    if norm > 1e-8:
        pooled = pooled / norm
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--model-name", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-npz", type=Path, default=Path("outputs/mert_external_embeddings.npz"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/mert_external_embeddings_manifest.csv"))
    args = parser.parse_args()

    files = discover_audio(args.input_dir)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    processor, model, torch = load_mert(args.model_name, args.device)
    embeddings = {}
    rows = []
    for idx, path in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] {path}")
        emb = embed_audio(path, processor, model, torch, args.sample_rate, args.device, args.max_seconds)
        key = f"emb_{idx:05d}"
        embeddings[key] = emb
        rows.append({"key": key, "file": path.as_posix(), "model": args.model_name, "dim": len(emb)})

    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz, **embeddings)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["key", "file", "model", "dim"])
        writer.writeheader()
        writer.writerows(rows)
    meta = {"model": args.model_name, "files": len(files), "sample_rate": args.sample_rate}
    args.out_npz.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote: {args.out_npz}")
    print(f"wrote: {args.out_csv}")


if __name__ == "__main__":
    main()
