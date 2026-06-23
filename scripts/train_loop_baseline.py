from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.features import extract_audio_features


DEFAULT_TARGETS = ["source_family", "reverb", "distortion", "filter", "stereo", "motion"]


def read_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/loop_synthetic"))
    parser.add_argument("--out", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    args = parser.parse_args()
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]

    rows = read_rows(args.dataset / "metadata.jsonl")
    if len(rows) < 20:
        raise SystemExit("Need at least 20 samples. Generate more loop data first.")

    X = np.vstack([extract_audio_features(args.dataset / row["file"]) for row in rows])
    train_idx, test_idx = train_test_split(np.arange(len(rows)), test_size=0.25, random_state=17)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    models = {}
    encoders = {}
    for target in targets:
        missing = [idx for idx, row in enumerate(rows) if target not in row]
        if missing:
            raise SystemExit(f"Target '{target}' is missing from metadata. Regenerate the dataset or choose other --targets.")
        encoder = LabelEncoder()
        y = encoder.fit_transform([row[target] for row in rows])
        model = RandomForestClassifier(n_estimators=220, random_state=17, class_weight="balanced")
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)

        print(f"\n[{target}]")
        print(classification_report(y[test_idx], pred, target_names=encoder.classes_, zero_division=0))

        models[target] = model
        encoders[target] = encoder

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "models": models, "encoders": encoders, "targets": targets}, args.out)
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
