from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".mp3", ".m4a"}


def safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


def run(cmd: list[str]) -> None:
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/external_loops"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/external_clap"))
    parser.add_argument("--prompts", type=Path, default=Path("configs/clap_prompts.json"))
    args = parser.parse_args()

    files = sorted(path for path in args.input_dir.iterdir() if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise SystemExit(f"No audio files found in {args.input_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        name = safe_stem(path)
        json_path = args.out_dir / f"{name}_clap.json"
        html_path = args.out_dir / f"{name}_clap.html"
        run([sys.executable, "scripts/clap_score_audio.py", "--audio", str(path), "--prompts", str(args.prompts), "--out", str(json_path)])
        run([sys.executable, "scripts/render_clap_report.py", "--input", str(json_path), "--out-html", str(html_path)])
        rows.append((path.name, json_path.name, html_path.name))

    index = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>External CLAP Batch</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:900px;margin:32px auto;color:#161616}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px;text-align:left}th{background:#f2f2f2}</style>",
        "<h1>External CLAP Batch</h1>",
        "<table><thead><tr><th>Audio</th><th>JSON</th><th>HTML</th></tr></thead><tbody>",
    ]
    for audio, json_name, html_name in rows:
        index.append(f"<tr><td>{audio}</td><td>{json_name}</td><td><a href='{html_name}'>CLAP report</a></td></tr>")
    index.append("</tbody></table>")
    (args.out_dir / "index.html").write_text("\n".join(index), encoding="utf-8")
    print(f"index: {args.out_dir / 'index.html'}")


if __name__ == "__main__":
    main()

