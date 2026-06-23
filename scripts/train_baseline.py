from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sound_timeline.features import extract_audio_features


def read_rows(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--out", type=Path, default=Path("models/baseline.joblib"))
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")
    if len(rows) < 10:
        raise SystemExit("Need at least 10 samples. Generate more data first.")

    features = []
    for row in rows:
        features.append(extract_audio_features(args.dataset / row["file"]))
    X = np.vstack(features)

    source_encoder = LabelEncoder()
    y_source = source_encoder.fit_transform([row["source"] for row in rows])

    effect_encoder = MultiLabelBinarizer()
    y_effects = effect_encoder.fit_transform([row["effects"] for row in rows])

    spatial_encoder = MultiLabelBinarizer()
    y_spatial = spatial_encoder.fit_transform([row["spatial_texture"] for row in rows])

    indices = np.arange(len(rows))
    source_counts = Counter(y_source)
    can_stratify = len(rows) >= 20 and min(source_counts.values()) >= 2
    stratify = y_source if can_stratify else None
    if not can_stratify:
        print("warning: dataset is too small or imbalanced for stratified split; using a random split")
    train_idx, test_idx = train_test_split(indices, test_size=0.25, random_state=7, stratify=stratify)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    source_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    source_model.fit(X_train, y_source[train_idx])
    source_pred = source_model.predict(X_test)

    effects_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    effects_model.fit(X_train, y_effects[train_idx])
    effects_pred = effects_model.predict(X_test)

    spatial_model = OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced"))
    spatial_model.fit(X_train, y_spatial[train_idx])
    spatial_pred = spatial_model.predict(X_test)

    print("\n[source]")
    print(classification_report(y_source[test_idx], source_pred, target_names=source_encoder.classes_, zero_division=0))

    print("\n[effects]")
    print(classification_report(y_effects[test_idx], effects_pred, target_names=effect_encoder.classes_, zero_division=0))

    print("\n[spatial_texture]")
    print(classification_report(y_spatial[test_idx], spatial_pred, target_names=spatial_encoder.classes_, zero_division=0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "source_encoder": source_encoder,
            "effect_encoder": effect_encoder,
            "spatial_encoder": spatial_encoder,
            "source_model": source_model,
            "effects_model": effects_model,
            "spatial_model": spatial_model,
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
