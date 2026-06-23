from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frames = []
    all_columns: list[str] = []
    for path in args.inputs:
        frame = pd.read_csv(path)
        frame["feature_source_table"] = path.as_posix()
        frames.append(frame)
        for column in frame.columns:
            if column not in all_columns:
                all_columns.append(column)
    merged = pd.concat([frame.reindex(columns=all_columns) for frame in frames], ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"rows: {len(merged)}")
    for path, frame in zip(args.inputs, frames):
        print(f"  {path}: {len(frame)}")
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
