import argparse
import csv
import html
from collections import defaultdict
from pathlib import Path


FEATURES = [
    "centroid",
    "rolloff",
    "bandwidth",
    "flatness",
    "zcr",
    "width",
    "rms_std",
    "rms_range",
    "motion_strength",
    "motion_freq",
]


def read_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            for key in FEATURES + ["start", "end"]:
                row[key] = float(row[key])
            row["segment_index"] = int(row["segment_index"])
            rows.append(row)
    return rows


def corpus_stats(rows):
    stats = {}
    for key in FEATURES:
        values = [row[key] for row in rows]
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        stats[key] = (mean, var ** 0.5 + 1e-8)
    return stats


def z(value, key, stats):
    mean, std = stats[key]
    return (value - mean) / std


def bucket(value, low=-0.45, high=0.45):
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "mid"


def trait_phrases(means, stats):
    centroid_z = z(means["centroid"], "centroid", stats)
    rolloff_z = z(means["rolloff"], "rolloff", stats)
    flatness_z = z(means["flatness"], "flatness", stats)
    zcr_z = z(means["zcr"], "zcr", stats)
    width_z = z(means["width"], "width", stats)
    rms_range_z = z(means["rms_range"], "rms_range", stats)
    motion_z = z(means["motion_strength"], "motion_strength", stats)

    tone = "bright" if max(centroid_z, rolloff_z) > 0.55 else "dark" if min(centroid_z, rolloff_z) < -0.55 else "mid-toned"
    texture = "grainy/noisy" if max(flatness_z, zcr_z) > 0.45 else "cleaner/smoother" if max(flatness_z, zcr_z) < -0.55 else "lightly textured"
    space = "wide/spread" if width_z > 0.5 else "centered/narrow" if width_z < -0.45 else "moderately wide"
    motion = "pumping or strongly moving" if max(rms_range_z, motion_z) > 0.55 else "steady/held" if max(rms_range_z, motion_z) < -0.45 else "subtle movement"
    return tone, texture, space, motion


def sound_hypothesis(tone, texture, space, motion):
    if "bright" in tone and "grainy" in texture and "moving" in motion:
        return "bright crushed rhythmic sample / digital synth-texture"
    if "dark" in tone and "moving" in motion and "centered" in space:
        return "dark ducked low loop / pumping muted layer"
    if "dark" in tone and "wide" in space:
        return "wide dark pad, processed vocal bed, or washed sampled texture"
    if "centered" in space and "steady" in motion:
        return "centered muted loop / restrained backing layer"
    if "bright" in tone and "wide" in space:
        return "wide bright glitch or widened high texture"
    if "grainy" in texture:
        return "muffled grainy sampled bed / noisy texture layer"
    return "hybrid processed loop / ambiguous modern pop texture"


def transition_summary(rows, stats):
    if len(rows) < 2:
        return "single segment"
    first = rows[0]
    last = rows[-1]
    changes = []
    for key, label in [
        ("centroid", "brightness"),
        ("flatness", "grain/noise"),
        ("width", "stereo width"),
        ("motion_strength", "motion"),
        ("rms_range", "dynamic range"),
    ]:
        delta = z(last[key], key, stats) - z(first[key], key, stats)
        if delta > 0.55:
            changes.append(f"{label} increases")
        elif delta < -0.55:
            changes.append(f"{label} decreases")
    return "; ".join(changes) if changes else "mostly consistent across the clip"


def summarize_clip(rows, stats):
    means = {key: sum(row[key] for row in rows) / len(rows) for key in FEATURES}
    tone, texture, space, motion = trait_phrases(means, stats)
    return {
        "file": rows[0]["file"],
        "segments": len(rows),
        "duration": f"{rows[0]['start']:.2f}-{rows[-1]['end']:.2f}s",
        "tone": tone,
        "texture": texture,
        "space": space,
        "motion": motion,
        "hypothesis": sound_hypothesis(tone, texture, space, motion),
        "transition": transition_summary(rows, stats),
        "mean_features": means,
    }


def segment_read(row, stats):
    means = {key: row[key] for key in FEATURES}
    tone, texture, space, motion = trait_phrases(means, stats)
    return {
        "time": f"{row['start']:.2f}-{row['end']:.2f}s",
        "read": f"{tone}, {texture}, {space}, {motion}",
        "hypothesis": sound_hypothesis(tone, texture, space, motion),
    }


def write_csv(cards, out):
    fieldnames = ["file", "duration", "hypothesis", "tone", "texture", "space", "motion", "transition", "segments"]
    with Path(out).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for card in cards:
            writer.writerow({key: card[key] for key in fieldnames})


def write_html(cards, by_file, stats, out):
    sections = []
    for card in cards:
        rows = []
        for row in by_file[card["file"]]:
            seg = segment_read(row, stats)
            rows.append(
                "<tr>"
                f"<td>{html.escape(seg['time'])}</td>"
                f"<td>{html.escape(seg['hypothesis'])}</td>"
                f"<td>{html.escape(seg['read'])}</td>"
                "</tr>"
            )
        sections.append(
            f"<section><h2>{html.escape(card['file'])}</h2>"
            f"<p><strong>Main read:</strong> {html.escape(card['hypothesis'])}</p>"
            f"<p><strong>Traits:</strong> {html.escape(card['tone'])} / {html.escape(card['texture'])} / {html.escape(card['space'])} / {html.escape(card['motion'])}</p>"
            f"<p><strong>Timeline behavior:</strong> {html.escape(card['transition'])}</p>"
            "<table><thead><tr><th>Time</th><th>Segment Read</th><th>Trait Mix</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></section>"
        )

    style = """
    body{font-family:Arial,sans-serif;max-width:1180px;margin:32px auto;color:#111}
    section{border-top:2px solid #111;padding-top:18px;margin-top:28px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0 24px}
    th,td{border:1px solid #ddd;padding:8px;text-align:left;vertical-align:top}
    th{background:#eee}
    p{line-height:1.45}
    """
    doc = f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>Individual Sound Cards</title><style>{style}</style></head>
<body>
<h1>Individual Sound Cards</h1>
<p>Each card describes the clip itself. These are not sameness labels; similar names can still refer to different sounds.</p>
{''.join(sections)}
</body>
</html>
"""
    Path(out).write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--filter-stem-prefix", default=None)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-html", required=True)
    args = parser.parse_args()

    rows = read_rows(args.features)
    if args.filter_stem_prefix:
        rows = [row for row in rows if row["stem"].startswith(args.filter_stem_prefix)]
    stats = corpus_stats(rows)
    by_file = defaultdict(list)
    for row in sorted(rows, key=lambda r: (r["file"], r["start"])):
        by_file[row["file"]].append(row)
    cards = [summarize_clip(file_rows, stats) for _, file_rows in sorted(by_file.items())]
    write_csv(cards, args.out_csv)
    write_html(cards, by_file, stats, args.out_html)
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
