import argparse
import collections
import csv
from pathlib import Path


WATCH_LABELS = [
    "granular_texture",
    "bitcrushed_synth_lead",
    "wavetable_noise",
    "fuzzy_lofi_synth",
    "digital_synth_lead",
    "supersaw_stack",
    "synth_pad_wash",
    "synth_pluck_bell",
    "arpeggio_sequence",
    "sidechained_synth_bass",
    "sub_808_synth_bass",
    "formant_vocoder",
]


def read_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summarize(path):
    rows = read_rows(path)
    usable = [row for row in rows if row.get("final_label") != "ambiguous"]
    return {
        "rows": len(rows),
        "usable": len(usable),
        "decision": collections.Counter(row.get("decision", "") for row in rows),
        "support": collections.Counter(row.get("source_kind_support", "") for row in rows),
        "labels": collections.Counter(row.get("final_label", "") for row in usable),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    args = parser.parse_args()

    for item in args.files:
        if "=" in item:
            name, raw_path = item.split("=", 1)
        else:
            raw_path = item
            name = Path(raw_path).stem
        summary = summarize(raw_path)
        print(f"\n## {name}")
        print(f"rows: {summary['rows']}")
        print(f"usable: {summary['usable']}")
        print("decisions:", dict(summary["decision"].most_common()))
        print("source_kind_support:", dict(summary["support"].most_common()))
        print("selected_labels:")
        for label in WATCH_LABELS:
            print(f"  {label}: {summary['labels'].get(label, 0)}")
        print("top_labels:", summary["labels"].most_common(12))


if __name__ == "__main__":
    main()
