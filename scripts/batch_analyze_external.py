from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--model", type=Path, default=Path("models/loop_baseline.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/external_batch"))
    parser.add_argument("--segment-seconds", type=float, default=4.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    args = parser.parse_args()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for audio_path in files:
        name = safe_stem(audio_path)
        json_path = args.out_dir / f"{name}_segments.json"
        segments_html = args.out_dir / f"{name}_segments.html"
        pseudo_csv = args.out_dir / f"{name}_pseudo_labels.csv"
        pseudo_html = args.out_dir / f"{name}_pseudo_labels.html"

        run(
            [
                sys.executable,
                "scripts/infer_audio_segments.py",
                "--model",
                str(args.model),
                "--audio",
                str(audio_path),
                "--segment-seconds",
                str(args.segment_seconds),
                "--hop-seconds",
                str(args.hop_seconds),
                "--out",
                str(json_path),
            ]
        )
        run([sys.executable, "scripts/render_segments_report.py", "--input", str(json_path), "--out-html", str(segments_html)])
        run([sys.executable, "scripts/pseudo_label_segments.py", "--input", str(json_path), "--out", str(pseudo_csv)])
        run([sys.executable, "scripts/render_pseudo_labels.py", "--input", str(pseudo_csv), "--out-html", str(pseudo_html)])

        index_rows.append((audio_path.name, segments_html.name, pseudo_html.name))

    index_html = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>External Clip Batch Analysis</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f2f2f2}</style>",
        "<h1>External Clip Batch Analysis</h1>",
        "<table><thead><tr><th>Audio</th><th>Segment Profile</th><th>Pseudo Labels</th></tr></thead><tbody>",
    ]
    for audio_name, segments_name, pseudo_name in index_rows:
        index_html.append(
            f"<tr><td>{audio_name}</td><td><a href='{segments_name}'>segment profile</a></td><td><a href='{pseudo_name}'>pseudo labels</a></td></tr>"
        )
    index_html.append("</tbody></table>")
    index_path = args.out_dir / "index.html"
    index_path.write_text("\n".join(index_html), encoding="utf-8")
    print(f"index: {index_path}")


if __name__ == "__main__":
    main()

