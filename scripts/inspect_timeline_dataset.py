from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


LANES = [
    "main_melody",
    "background_texture",
    "bass_foundation",
    "transition_fx",
    "rhythmic_layer",
]

ROLE_COLORS = {
    "main_melody": "#d95f02",
    "background_texture": "#1b9e77",
    "bass_foundation": "#7570b3",
    "transition_fx": "#e7298a",
    "rhythmic_layer": "#66a61e",
}


def read_rows(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_spectrogram(audio_path: Path, out_path: Path) -> None:
    import librosa
    import librosa.display

    audio, sr = sf.read(audio_path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=96, fmax=12_000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=12_000, ax=ax)
    ax.set_title(audio_path.name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def region_html(row: dict) -> str:
    duration = float(row["duration"])
    lane_height = 42
    chart_height = len(LANES) * lane_height
    region_divs = []

    for region in row["regions"]:
        role = region["role"]
        lane_idx = LANES.index(role) if role in LANES else len(LANES) - 1
        left = 100.0 * float(region["start"]) / duration
        width = 100.0 * (float(region["end"]) - float(region["start"])) / duration
        top = lane_idx * lane_height + 6
        color = ROLE_COLORS.get(role, "#555")
        label = f"{region['source']} | {', '.join(region['effects'])}"
        title = json.dumps(region, ensure_ascii=False)
        region_divs.append(
            f"<div class='region' title='{title}' style='left:{left:.3f}%;width:{width:.3f}%;top:{top}px;background:{color};'>"
            f"<span>{label}</span></div>"
        )

    lane_labels = []
    for idx, lane in enumerate(LANES):
        top = idx * lane_height
        lane_labels.append(f"<div class='lane-label' style='top:{top}px'>{lane}</div>")
        lane_labels.append(f"<div class='lane-line' style='top:{top + lane_height - 1}px'></div>")

    return f"<div class='timeline' style='height:{chart_height}px'>{''.join(lane_labels)}{''.join(region_divs)}</div>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/timeline_synthetic"))
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()

    rows = read_rows(args.dataset / "metadata.jsonl")[: args.count]
    preview_dir = args.dataset / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Timeline Synthetic Dataset Preview</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;max-width:1120px;margin:24px auto;color:#161616}",
        ".item{border-top:1px solid #ddd;padding:18px 0 28px}",
        "audio{width:100%;margin:8px 0 12px}",
        "img{max-width:100%;border:1px solid #ddd}",
        ".timeline{position:relative;margin:12px 0 16px 150px;border:1px solid #ccc;background:#fafafa}",
        ".lane-label{position:absolute;left:-145px;width:135px;font-size:12px;color:#333;text-align:right;padding-top:13px}",
        ".lane-line{position:absolute;left:0;right:0;height:1px;background:#e6e6e6}",
        ".region{position:absolute;height:28px;border-radius:4px;color:white;font-size:11px;overflow:hidden;box-sizing:border-box;border:1px solid rgba(0,0,0,.18)}",
        ".region span{display:block;padding:7px 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}",
        "code{display:block;white-space:pre-wrap;background:#f5f5f5;padding:10px;border:1px solid #ddd;font-size:12px}",
        "</style>",
        "<h1>Timeline Synthetic Dataset Preview</h1>",
    ]

    for row in rows:
        audio_path = args.dataset / row["file"]
        png_name = Path(row["file"]).with_suffix(".png").name
        render_spectrogram(audio_path, preview_dir / png_name)

        html_lines.extend(
            [
                "<div class='item'>",
                f"<h2>{audio_path.name}</h2>",
                f"<audio controls src='{row['file']}'></audio>",
                region_html(row),
                f"<p><img src='previews/{png_name}'></p>",
                f"<code>{json.dumps(row, ensure_ascii=False, indent=2)}</code>",
                "</div>",
            ]
        )

    out_path = args.dataset / "timeline_preview.html"
    out_path.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"preview: {out_path}")


if __name__ == "__main__":
    main()

