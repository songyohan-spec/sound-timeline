from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.features import extract_audio_features


def top_predictions(model, encoder, x, top_k: int) -> list[dict]:
    if not hasattr(model, "predict_proba"):
        label = encoder.inverse_transform(model.predict(x))[0]
        return [{"label": str(label), "confidence": None}]

    probs = model.predict_proba(x)[0]
    order = probs.argsort()[::-1][:top_k]
    labels = encoder.inverse_transform(order)
    return [
        {
            "label": str(label),
            "confidence": round(float(probs[index]), 4),
        }
        for label, index in zip(labels, order)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.audio.exists():
        raise SystemExit(f"Audio not found: {args.audio}")

    bundle = joblib.load(args.model)
    scaler = bundle["scaler"]
    models = bundle["models"]
    encoders = bundle["encoders"]
    targets = bundle["targets"]

    features = extract_audio_features(args.audio)
    x = scaler.transform([features])

    predictions = {}
    for target in targets:
        predictions[target] = top_predictions(models[target], encoders[target], x, args.top_k)

    report = {
        "audio": str(args.audio),
        "model": str(args.model),
        "predictions": predictions,
        "caution": "These are categorical sound-profile estimates, not exact plugin parameters or effect-chain order.",
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()

