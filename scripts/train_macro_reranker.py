from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


TARGETS = ["target_source_macro", "target_processing_macro"]


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.startswith("score::")]


def print_resubstitution_report(x: np.ndarray, frame: pd.DataFrame, target: str) -> tuple[RandomForestClassifier, LabelEncoder]:
    encoder = LabelEncoder()
    y = encoder.fit_transform(frame[target].astype(str))
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=31,
        class_weight="balanced",
        min_samples_leaf=1,
    )
    model.fit(x, y)
    pred = model.predict(x)
    print(f"\n[{target} / train-set sanity check]")
    print(classification_report(y, pred, target_names=encoder.classes_, zero_division=0))
    return model, encoder


def print_leave_one_timeline_report(x: np.ndarray, frame: pd.DataFrame, target: str) -> dict:
    timelines = sorted(frame["timeline"].astype(str).unique())
    if len(timelines) < 2:
        print(f"\n[{target} / leave-one-timeline] skipped: need at least 2 timelines")
        return {"accuracy": 0.0, "macro_f1": 0.0}

    y_labels = frame[target].astype(str).to_numpy()
    predictions: list[str] = []
    truths: list[str] = []
    for timeline in timelines:
        test_mask = frame["timeline"].astype(str).to_numpy() == timeline
        train_mask = ~test_mask
        train_labels = sorted(set(y_labels[train_mask]))
        if len(train_labels) < 2:
            majority = max(set(y_labels[train_mask]), key=list(y_labels[train_mask]).count)
            predictions.extend([majority] * int(test_mask.sum()))
            truths.extend(y_labels[test_mask])
            continue
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_labels[train_mask])
        model = RandomForestClassifier(
            n_estimators=220,
            random_state=31,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        model.fit(x[train_mask], y_train)
        pred = encoder.inverse_transform(model.predict(x[test_mask]))
        predictions.extend(pred.tolist())
        truths.extend(y_labels[test_mask].tolist())

    labels = sorted(set(truths) | set(predictions))
    print(f"\n[{target} / leave-one-timeline estimate]")
    print(classification_report(truths, predictions, labels=labels, zero_division=0))
    accuracy = float(np.mean(np.array(truths) == np.array(predictions)))
    macro_f1 = float(f1_score(truths, predictions, labels=labels, average="macro", zero_division=0))
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("outputs/macro_reranker_dataset.csv"))
    parser.add_argument("--out", type=Path, default=Path("models/macro_reranker.joblib"))
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset, encoding="utf-8-sig").fillna(0.0)
    features = feature_columns(frame)
    if len(frame) < 8:
        raise SystemExit("Need at least 8 rows for this baseline.")
    if not features:
        raise SystemExit("No score:: feature columns found.")

    x = frame[features].astype(float).to_numpy()
    models = {}
    encoders = {}
    validation = {}
    confidence_caps = {}
    for target in TARGETS:
        if target not in frame.columns:
            raise SystemExit(f"Missing target column: {target}")
        validation[target] = print_leave_one_timeline_report(x, frame, target)
        confidence_caps[target] = round(max(0.5, validation[target]["macro_f1"]), 4)
        model, encoder = print_resubstitution_report(x, frame, target)
        models[target] = model
        encoders[target] = encoder

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "features": features,
            "models": models,
            "encoders": encoders,
            "targets": TARGETS,
            "validation": validation,
            "confidence_caps": confidence_caps,
            "note": "Tiny baseline trained on pseudo-calibrated CLAP palette timelines. Treat as a reranker prototype, not a validated model.",
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
