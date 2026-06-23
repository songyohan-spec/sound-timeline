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


def read_rows(metadata_path: Path) -> list[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_group_key(row: dict) -> str:
    source = str(row.get("source_file", ""))
    # Split by original source file so duplicate crops/copies do not leak
    # across train/test when the same source was bootstrapped into a label.
    return source or row["file"]


def split_by_source(rows: list[dict], test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = sorted({source_group_key(row) for row in rows})
    train_groups, test_groups = train_test_split(groups, test_size=test_size, random_state=seed)
    train_set = set(train_groups)
    train_idx = np.array([idx for idx, row in enumerate(rows) if source_group_key(row) in train_set])
    test_idx = np.array([idx for idx, row in enumerate(rows) if source_group_key(row) not in train_set])
    return train_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/reference_element_dataset_bootstrap"))
    parser.add_argument("--out", type=Path, default=Path("models/reference_element_baseline.joblib"))
    parser.add_argument("--targets", default="group,label")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")
    if len(rows) < 20:
        raise SystemExit("Need at least 20 samples.")

    targets = [target.strip() for target in args.targets.split(",") if target.strip()]
    X = np.vstack([extract_audio_features(args.dataset / row["file"]) for row in rows])
    train_idx, test_idx = split_by_source(rows, args.test_size, args.seed)
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise SystemExit("Train/test split failed. Add more distinct source files.")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    models = {}
    encoders = {}
    for target in targets:
        encoder = LabelEncoder()
        y = encoder.fit_transform([row[target] for row in rows])
        model = RandomForestClassifier(
            n_estimators=260,
            random_state=args.seed,
            class_weight="balanced_subsample",
            max_features="sqrt",
        )
        model.fit(X_train, y[train_idx])
        pred = model.predict(X_test)
        print(f"\n[{target}]")
        print(classification_report(y[test_idx], pred, labels=np.unique(y[test_idx]), target_names=encoder.inverse_transform(np.unique(y[test_idx])), zero_division=0))
        models[target] = model
        encoders[target] = encoder

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scaler": scaler,
            "models": models,
            "encoders": encoders,
            "targets": targets,
            "dataset": str(args.dataset),
            "warning": "Bootstrap data may contain copied source files under multiple labels. Use this model only as a pipeline check until real per-label samples are collected.",
        },
        args.out,
    )
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
