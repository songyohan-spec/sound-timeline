from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


META_COLUMNS = {
    "file",
    "label",
    "group",
    "priority",
    "source_file",
    "source_identity",
    "source_index",
    "duration",
}


def split_by_identity(df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    groups = df["source_identity"].fillna(df["source_file"]).astype(str).to_numpy()
    indices = np.arange(len(df))
    train_idx, test_idx = next(splitter.split(indices, groups=groups))
    return train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/reference_element_features_bootstrap.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/reference_element_fast_bootstrap.joblib"))
    parser.add_argument("--targets", default="group,label")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if len(df) < 20:
        raise SystemExit("Need at least 20 feature rows.")

    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    train_idx, test_idx = split_by_identity(df, args.test_size, args.seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("Train/test split failed. Add more source identities.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    models = {}
    encoders = {}
    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    for target in targets:
        encoder = LabelEncoder()
        y = encoder.fit_transform(df[target].astype(str).to_numpy())
        train_labels = set(y[train_idx])
        test_labels = sorted(set(y[test_idx]) & train_labels)
        if not test_labels:
            print(f"\n[{target}] skipped: no overlapping train/test labels")
            continue

        model = RandomForestClassifier(
            n_estimators=400,
            random_state=args.seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        print(f"\n[{target}]")
        print(
            classification_report(
                y[test_idx],
                pred,
                labels=np.array(test_labels),
                target_names=encoder.inverse_transform(np.array(test_labels)),
                zero_division=0,
            )
        )
        models[target] = model
        encoders[target] = encoder

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "models": models,
            "encoders": encoders,
            "targets": list(models.keys()),
            "feature_cols": feature_cols,
            "features": str(args.features),
            "warning": (
                "This bootstrap model is an infrastructure check. The source audio was copied "
                "from broad dry folders into semantic labels, so label accuracy is not evidence "
                "of real sound-element recognition."
            ),
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
