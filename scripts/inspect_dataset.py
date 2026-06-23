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


def read_rows(metadata_path: Path) -> list[dict]:
    rows = []
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def render_preview(audio_path: Path, out_path: Path) -> None:
    import librosa
    import librosa.display

    audio, sr = sf.read(audio_path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=96, fmax=12_000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=12_000, ax=ax)
    ax.set_title(audio_path.name)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    metadata_path = args.dataset / "metadata.jsonl"
    preview_dir = args.dataset / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(metadata_path)[: args.count]
    html_lines = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Sound Timeline Dataset Preview</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:980px;margin:24px auto} .item{border-top:1px solid #ddd;padding:16px 0} img{max-width:100%} code{white-space:pre-wrap}</style>",
        "<h1>Sound Timeline Dataset Preview</h1>",
    ]

    for row in rows:
        audio_rel = row["file"]
        audio_path = args.dataset / audio_rel
        png_name = Path(audio_rel).with_suffix(".png").name
        png_path = preview_dir / png_name
        render_preview(audio_path, png_path)

        html_lines.extend(
            [
                "<div class='item'>",
                f"<h2>{audio_path.name}</h2>",
                f"<audio controls src='{audio_rel.replace(chr(92), '/')}'></audio>",
                f"<p><img src='previews/{png_name}'></p>",
                f"<code>{json.dumps(row, ensure_ascii=False, indent=2)}</code>",
                "</div>",
            ]
        )

    index_path = args.dataset / "preview.html"
    index_path.write_text("\n".join(html_lines), encoding="utf-8")
    print(f"preview: {index_path}")


if __name__ == "__main__":
    main()

