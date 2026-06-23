from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


SYNTHISH_LAYERS = {"synth"}
SYNTHISH_LABEL_HINTS = (
    "synth",
    "vocoder",
    "formant",
    "808",
    "sidechained_bass",
    "sidechained_synth_bass",
)


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def is_synthish_region(row: dict, include_bass: bool, include_vocal_hybrid: bool) -> bool:
    layer = row.get("layer", "")
    label = row.get("label", "").lower()
    if layer in SYNTHISH_LAYERS:
        return True
    if include_bass and layer == "bass" and any(hint in label for hint in SYNTHISH_LABEL_HINTS):
        return True
    if include_vocal_hybrid and layer == "vocals" and any(hint in label for hint in SYNTHISH_LABEL_HINTS):
        return True
    return False


def load_stem(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float32)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio[:, :2], sr


def apply_region(mask: np.ndarray, start: float, end: float, sr: int, fade_seconds: float) -> None:
    start_i = max(0, min(len(mask), int(round(start * sr))))
    end_i = max(start_i, min(len(mask), int(round(end * sr))))
    if end_i <= start_i:
        return
    mask[start_i:end_i] = 1.0
    fade = max(1, int(round(fade_seconds * sr)))
    fade = min(fade, max(1, (end_i - start_i) // 2))
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        mask[start_i : start_i + fade] = np.maximum(mask[start_i : start_i + fade], ramp)
        mask[end_i - fade : end_i] = np.maximum(mask[end_i - fade : end_i], ramp[::-1])


def normalize_peak(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    max_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    if max_abs > peak:
        return audio * (peak / max_abs)
    return audio


def stem_path(stems_root: Path, track: str, stem: str) -> Path:
    return stems_root / track / f"{stem}.wav"


def write_html(summary_rows: list[dict], out_html: Path) -> None:
    by_track = sorted(summary_rows, key=lambda row: row["track"])
    layer_counts = Counter()
    label_counts = Counter()
    for row in summary_rows:
        for item in row["labels"].split("|"):
            if item:
                label_counts[item] += 1
        for item in row["layers"].split("|"):
            if item:
                layer_counts[item] += 1

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    trs = []
    for row in by_track:
        synth_rel = Path(os.path.relpath(row["synth_candidate"], out_html.parent)).as_posix()
        resid_rel = Path(os.path.relpath(row["residual_context"], out_html.parent)).as_posix()
        mix_rel = Path(os.path.relpath(row["stem_mix"], out_html.parent)).as_posix()
        trs.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td>{row['region_count']}</td>"
            f"<td>{html.escape(row['layers'])}<br><small>{html.escape(row['labels'])}</small></td>"
            f"<td><audio controls src='{html.escape(mix_rel)}'></audio></td>"
            f"<td><audio controls src='{html.escape(synth_rel)}'></audio></td>"
            f"<td><audio controls src='{html.escape(resid_rel)}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Candidate Audio Exports</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 260px; }}
small {{ color: #555; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Synth Candidate Audio Exports</h1>
<p class="note">Pseudo-separation generated from Demucs stems and context-aware synth regions. This is not a true isolated synth stem; it is an auditionable candidate layer used to check whether the model is pulling the right sound family.</p>
{count_table("Exported Layers", layer_counts)}
{count_table("Exported Labels", label_counts)}
<section>
<h2>Tracks</h2>
<table>
<tr><th>Track</th><th>Regions</th><th>Labels</th><th>Stem Mix</th><th>Synth Candidate</th><th>Residual Context</th></tr>
{''.join(trs)}
</table>
</section>
</html>"""
    out_html.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, default=Path("outputs/demucs_stems_6s_full/broad_layer_regions_likely_multiscale.csv"))
    parser.add_argument("--stems-root", type=Path, default=Path("outputs/demucs_stems_6s_full/htdemucs_6s"))
    parser.add_argument("--out-root", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio"))
    parser.add_argument("--include-bass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-vocal-hybrid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fade-seconds", type=float, default=0.04)
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_index.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_index.html"))
    args = parser.parse_args()

    region_rows = [
        row
        for row in read_rows(args.regions)
        if is_synthish_region(row, include_bass=args.include_bass, include_vocal_hybrid=args.include_vocal_hybrid)
    ]
    by_track: dict[str, list[dict]] = defaultdict(list)
    for row in region_rows:
        by_track[row["track"]].append(row)

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    missing = []
    for track, rows in sorted(by_track.items()):
        stem_names = sorted({stem for row in rows for stem in row.get("stems", "").split("|") if stem})
        loaded = {}
        sr = None
        target_len = 0
        for stem in stem_names:
            path = stem_path(args.stems_root, track, stem)
            if not path.exists():
                missing.append(str(path))
                continue
            audio, stem_sr = load_stem(path)
            if sr is None:
                sr = stem_sr
            if stem_sr != sr:
                raise SystemExit(f"Sample-rate mismatch in {track}: {stem_sr} vs {sr}")
            loaded[stem] = audio
            target_len = max(target_len, len(audio))
        if not loaded or sr is None:
            continue

        synth = np.zeros((target_len, 2), dtype=np.float32)
        mix = np.zeros_like(synth)
        active_masks: dict[str, np.ndarray] = {stem: np.zeros(target_len, dtype=np.float32) for stem in loaded}
        for stem, audio in loaded.items():
            mix[: len(audio)] += audio
        for row in rows:
            start = safe_float(row["start"])
            end = safe_float(row["end"])
            for stem in row.get("stems", "").split("|"):
                stem = stem.strip()
                if stem in active_masks:
                    apply_region(active_masks[stem], start, end, sr, args.fade_seconds)
        for stem, audio in loaded.items():
            mask = active_masks[stem][: len(audio), None]
            synth[: len(audio)] += audio * mask

        residual = mix - synth
        track_dir = args.out_root / track
        track_dir.mkdir(parents=True, exist_ok=True)
        mix_path = track_dir / "stem_mix.wav"
        synth_path = track_dir / "synth_candidate.wav"
        residual_path = track_dir / "residual_context.wav"
        sf.write(mix_path, normalize_peak(mix), sr)
        sf.write(synth_path, normalize_peak(synth), sr)
        sf.write(residual_path, normalize_peak(residual), sr)
        summary_rows.append(
            {
                "track": track,
                "region_count": str(len(rows)),
                "layers": "|".join(sorted({row["layer"] for row in rows})),
                "labels": "|".join(sorted({row["label"] for row in rows})),
                "stems": "|".join(stem_names),
                "stem_mix": str(mix_path),
                "synth_candidate": str(synth_path),
                "residual_context": str(residual_path),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["track", "region_count", "layers", "labels", "stems", "stem_mix", "synth_candidate", "residual_context"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    write_html(summary_rows, args.out_html)
    print(f"regions used: {len(region_rows)}")
    print(f"tracks exported: {len(summary_rows)}")
    if missing:
        print(f"missing stems: {len(missing)}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
