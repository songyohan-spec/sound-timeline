from __future__ import annotations

import argparse
import csv
import html
import os
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def load_audio(path: Path) -> np.ndarray:
    audio, _sr = sf.read(path, always_2d=True)
    return audio.astype(np.float32)


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio)) + 1e-12))


def db(value: float) -> float:
    return 20.0 * float(np.log10(max(value, 1e-12)))


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def classify(candidate_share: float, mix_db: float, candidate_db: float, residual_db: float) -> str:
    if mix_db < -45:
        return "silent_selection"
    if candidate_db < -45:
        return "empty_candidate"
    if candidate_share < 0.03:
        return "too_sparse"
    if candidate_share > 0.82:
        return "too_broad"
    if candidate_share > 0.65 and residual_db < candidate_db - 10:
        return "likely_overcaptures"
    return "auditionable"


def score_rows(index_rows: list[dict]) -> list[dict]:
    out = []
    for row in index_rows:
        mix_path = Path(row["stem_mix"])
        candidate_path = Path(row["synth_candidate"])
        residual_path = Path(row["residual_context"])
        mix = load_audio(mix_path)
        candidate = load_audio(candidate_path)
        residual = load_audio(residual_path)
        mix_rms = rms(mix)
        candidate_rms = rms(candidate)
        residual_rms = rms(residual)
        share = safe_ratio(candidate_rms, mix_rms)
        scored = dict(row)
        scored.update(
            {
                "mix_rms_db": f"{db(mix_rms):.2f}",
                "candidate_rms_db": f"{db(candidate_rms):.2f}",
                "residual_rms_db": f"{db(residual_rms):.2f}",
                "candidate_rms_share": f"{share:.4f}",
                "triage": classify(share, db(mix_rms), db(candidate_rms), db(residual_rms)),
            }
        )
        out.append(scored)
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_html(rows: list[dict], path: Path) -> None:
    triage_counts = Counter(row["triage"] for row in rows)

    def count_table(title: str, counter: Counter[str]) -> str:
        body = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in counter.most_common())
        return f"<section><h2>{html.escape(title)}</h2><table><tr><th>Item</th><th>Count</th></tr>{body}</table></section>"

    trs = []
    for row in sorted(rows, key=lambda r: (r["triage"], r["track"])):
        synth_rel = Path(os.path.relpath(row["synth_candidate"], path.parent)).as_posix()
        resid_rel = Path(os.path.relpath(row["residual_context"], path.parent)).as_posix()
        mix_rel = Path(os.path.relpath(row["stem_mix"], path.parent)).as_posix()
        trs.append(
            "<tr>"
            f"<td>{html.escape(row['track'])}</td>"
            f"<td class='{html.escape(row['triage'])}'><b>{html.escape(row['triage'])}</b></td>"
            f"<td>{html.escape(row['candidate_rms_share'])}</td>"
            f"<td>{html.escape(row['candidate_rms_db'])}</td>"
            f"<td>{html.escape(row['residual_rms_db'])}</td>"
            f"<td>{html.escape(row['labels'])}</td>"
            f"<td><audio controls src='{html.escape(mix_rel)}'></audio></td>"
            f"<td><audio controls src='{html.escape(synth_rel)}'></audio></td>"
            f"<td><audio controls src='{html.escape(resid_rel)}'></audio></td>"
            "</tr>"
        )

    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Synth Candidate Separation Triage</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111; }}
section {{ border-top: 2px solid #111; margin-top: 28px; padding-top: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d8d8d8; padding: 7px; vertical-align: top; text-align: left; }}
th {{ background: #f0f0f0; }}
audio {{ width: 220px; }}
.auditionable {{ background: #dff3e6; }}
.too_sparse {{ background: #fff2c8; }}
.too_broad, .likely_overcaptures, .empty_candidate, .silent_selection {{ background: #f7dede; }}
.note {{ color: #444; max-width: 1040px; }}
</style>
<h1>Synth Candidate Separation Triage</h1>
<p class="note">Sanity checks for pseudo-separated synth candidates. These scores are not ground truth. They only flag candidates that are too quiet, too sparse, or too broad before manual audition or model refinement.</p>
{count_table("Triage", triage_counts)}
<section>
<h2>Tracks</h2>
<table>
<tr><th>Track</th><th>Triage</th><th>Candidate/Mix RMS</th><th>Candidate dB</th><th>Residual dB</th><th>Labels</th><th>Stem Mix</th><th>Synth Candidate</th><th>Residual</th></tr>
{''.join(trs)}
</table>
</section>
</html>"""
    path.write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_strict_index.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/demucs_stems_6s_full/synth_candidate_audio_strict_triage.html"))
    args = parser.parse_args()

    rows = score_rows(read_rows(args.index))
    if not rows:
        raise SystemExit("No rows to score.")
    write_csv(rows, args.out_csv)
    write_html(rows, args.out_html)
    print(f"tracks scored: {len(rows)}")
    print(f"triage: {dict(Counter(row['triage'] for row in rows).most_common())}")
    print(f"wrote: {args.out_csv}")
    print(f"wrote: {args.out_html}")


if __name__ == "__main__":
    main()
