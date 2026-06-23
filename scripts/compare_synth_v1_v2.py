from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    for suffix in ["", "_v2", "_v2w"]:
        name = suffix or "_v1"
        rows = read_rows(Path(f"outputs/demucs_stems_6s_full/synth_specialist{suffix}.csv"))
        active = [row for row in rows if row["strength"] in {"medium", "strong"}]
        print(f"\nSPECIALIST{name} rows={len(rows)} active={len(active)}")
        print("strength", Counter(row["strength"] for row in rows).most_common())
        print("labels", Counter(row["synth_label_top"] for row in active).most_common(12))
        print("stems", Counter(row["stem"] for row in active).most_common())

    ensemble_files = [
        ("_v1", Path("outputs/demucs_stems_6s_full/synth_fast_ensemble.csv")),
        ("_v2", Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v2.csv")),
        ("_v2w", Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v2w.csv")),
        ("_v2w_strict", Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v2w_strict.csv")),
        ("_v3_strict", Path("outputs/demucs_stems_6s_full/synth_fast_ensemble_v3_strict.csv")),
    ]
    for name, path in ensemble_files:
        if not path.exists():
            continue
        rows = read_rows(path)
        usable = [row for row in rows if row["final_label"] != "ambiguous"]
        print(f"\nENSEMBLE{name} usable={len(usable)}")
        print("decisions", Counter(row["decision"] for row in rows).most_common())
        print("labels", Counter(row["final_label"] for row in usable).most_common(12))
        print("stems", Counter(row["stem"] for row in usable).most_common())


if __name__ == "__main__":
    main()
