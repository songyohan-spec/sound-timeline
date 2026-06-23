from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


META_COLUMNS = {"file", "label", "family", "base_label", "duration", "source_index", "dataset"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/synth_specialist_features_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/synth_specialist_v1.joblib"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=527)
    parser.add_argument("--pseudo-weight", type=float, default=0.35)
    args = parser.parse_args()

    frame = pd.read_csv(args.features)
    feature_cols = [col for col in frame.columns if col not in META_COLUMNS]
    X = frame[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_label = frame["label"].astype(str).to_numpy()
    y_family = frame["family"].astype(str).to_numpy()
    if "dataset" in frame.columns:
        weak_dataset = (
            frame["dataset"].astype(str).str.contains("pseudo", case=False, regex=False)
            | frame["dataset"].astype(str).str.contains("teacher", case=False, regex=False)
        ).to_numpy()
        sample_weight = np.where(
            weak_dataset,
            args.pseudo_weight,
            1.0,
        ).astype(np.float32)
    else:
        sample_weight = np.ones(len(frame), dtype=np.float32)

    train_idx, test_idx = train_test_split(
        np.arange(len(frame)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_label,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    outputs = {}
    for target_name, y in [("family", y_family), ("label", y_label)]:
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        model = RandomForestClassifier(
            n_estimators=520,
            random_state=args.seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(X_train, y_encoded[train_idx], sample_weight=sample_weight[train_idx])
        pred = model.predict(X_test)
        print(f"\n[{target_name}]")
        print(classification_report(y_encoded[test_idx], pred, target_names=encoder.classes_, zero_division=0))
        outputs[target_name] = {"model": model, "encoder": encoder}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "outputs": outputs,
            "features": str(args.features),
            "pseudo_weight": args.pseudo_weight,
            "warning": "Synth specialist trained on synthetic/hard-negative examples. Use as a focused hypothesis model, not exact oscillator truth.",
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
