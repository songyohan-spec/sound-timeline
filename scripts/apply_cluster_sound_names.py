import argparse
import csv
from pathlib import Path


BOLD_ALT_POP_NAMES = {
    0: {
        "human_label": "muffled_grainy_sample_bed",
        "notes": "Grainy but not very bright; likely reads as a muted sampled bed, noisy loop, or softened texture layer.",
    },
    1: {
        "human_label": "centered_muted_loop",
        "notes": "Narrow and low-motion; likely a centered muted loop or intimate low-energy layer.",
    },
    2: {
        "human_label": "wide_dark_pad_or_vocal_texture",
        "notes": "Dark and very wide; likely a washed pad, processed vocal bed, or wide resampled texture.",
    },
    3: {
        "human_label": "bright_crushed_pulsing_loop",
        "notes": "Bright, noisy, and dynamic; likely a crushed/pulsing sampled loop or aggressive texture layer.",
    },
    4: {
        "human_label": "wide_bright_glitch_texture",
        "notes": "Very bright, noisy, and wide; likely glitchy/wide digital texture or harsh widened layer.",
    },
    5: {
        "human_label": "dark_pumping_low_texture",
        "notes": "Dark, narrow, and strongly dynamic; likely pumping low texture, ducked loop, or dark rhythmic layer.",
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--preset", choices=["bold-alt-pop"], default="bold-alt-pop")
    args = parser.parse_args()

    mapping = BOLD_ALT_POP_NAMES
    rows = []
    with Path(args.input).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            cluster = int(row["cluster"])
            if cluster in mapping:
                row.update(mapping[cluster])
            rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
