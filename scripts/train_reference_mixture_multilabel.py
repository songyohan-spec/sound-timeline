from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


META_COLUMNS = {
    "file",
    "labels",
    "groups",
    "primary_label",
    "primary_group",
    "source_file",
    "source_index",
    "duration",
    "training_labels",
    "feature_source_table",
}


def split_pipe(value: str) -> list[str]:
    return [item for item in str(value).split("|") if item]


def positive_probabilities(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    columns = []
    probs_raw = model.predict_proba(x)
    for idx, raw in enumerate(probs_raw):
        classes = model.classes_[idx]
        if len(classes) == 1:
            columns.append(np.ones(len(x), dtype=np.float32) * float(classes[0]))
        else:
            one_index = int(np.where(classes == 1)[0][0]) if 1 in classes else len(classes) - 1
            columns.append(raw[:, one_index].astype(np.float32))
    return np.vstack(columns).T


def calibrate_thresholds(y_true: np.ndarray, y_prob: np.ndarray, class_names: np.ndarray) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    grid = np.round(np.arange(0.12, 0.76, 0.03), 2)
    for idx, name in enumerate(class_names):
        positives = y_true[:, idx]
        if positives.sum() == 0:
            thresholds[str(name)] = 0.5
            continue
        best_threshold = 0.5
        best_score = -1.0
        for threshold in grid:
            pred = (y_prob[:, idx] >= threshold).astype(int)
            tp = float(((pred == 1) & (positives == 1)).sum())
            fp = float(((pred == 1) & (positives == 0)).sum())
            fn = float(((pred == 0) & (positives == 1)).sum())
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
            # Mildly prefer stricter thresholds when F1 ties.
            score = f1 + (threshold * 0.002)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)
        thresholds[str(name)] = best_threshold
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/reference_mixture_features_v1.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/reference_mixture_multilabel_v1.joblib"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    feature_cols = [col for col in df.columns if col not in META_COLUMNS]
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    label_sets = [split_pipe(value) for value in df["labels"]]
    group_sets = [split_pipe(value) for value in df["groups"]]

    scaler = StandardScaler()
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=args.test_size, random_state=args.seed)
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    outputs = {}
    thresholds = {}
    for name, sets in [("groups", group_sets), ("labels", label_sets)]:
        binarizer = MultiLabelBinarizer()
        y = binarizer.fit_transform(sets)
        model = RandomForestClassifier(
            n_estimators=420,
            random_state=args.seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
        )
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        y_prob = positive_probabilities(model, X_test)
        thresholds[name] = calibrate_thresholds(y[test_idx], y_prob, binarizer.classes_)
        print(f"\n[{name}]")
        print(classification_report(y[test_idx], pred, target_names=binarizer.classes_, zero_division=0))
        print("[calibrated_thresholds]")
        for label, threshold in sorted(thresholds[name].items()):
            print(f"{label}: {threshold:.2f}")
        outputs[name] = {"model": model, "binarizer": binarizer}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "outputs": outputs,
            "thresholds": thresholds,
            "feature_cols": feature_cols,
            "features": str(args.features),
            "warning": "Synthetic mixture multi-label model. Useful for pipeline development; needs real reference data before claims.",
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
