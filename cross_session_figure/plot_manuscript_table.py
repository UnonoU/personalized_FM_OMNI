from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from plot_style import render_figure


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-input", type=Path, default=here / "manuscript_table_values.csv")
    parser.add_argument("--raw-statistics", type=Path, default=here / "cross_session_statistics.csv")
    parser.add_argument("--output-dir", type=Path, default=here / "reference_style_version")
    parser.add_argument("--style", choices=["classic", "reference"], default="reference")
    parser.add_argument("--subject-data", type=Path, default=here / "paired_subject_accuracy.csv")
    parser.add_argument("--no-subject-points", action="store_true")
    args = parser.parse_args()
    table = pd.read_csv(args.table_input)
    raw = pd.read_csv(args.raw_statistics)
    key = ["setting", "session", "method"]
    expected = {(setting, session, method) for setting in ("Multitask", "NBack")
                for session in (1, 2, 3) for method in ("Baseline", "Rest-LoRA", "Rest-Full")}
    for name, frame in (("manuscript table", table), ("raw statistics", raw)):
        if frame.duplicated(key).any() or set(frame[key].itertuples(index=False, name=None)) != expected:
            raise ValueError(f"{name}: all 18 unique conditions are required")
    table["mean"] = pd.to_numeric(table["mean"], errors="raise")
    table["SD"] = pd.to_numeric(table["SD"], errors="raise")
    if not np.isfinite(table[["mean", "SD"]]).all().all():
        raise ValueError("The manuscript mean/SD values must be finite")
    comparison = raw.rename(columns={"mean": "mean_from_subject_data", "SD": "SD_from_subject_data",
                                     "mean_gain_over_baseline": "gain_from_subject_data"})
    comparison = table.merge(comparison, on=key, validate="one_to_one", how="inner")
    comparison["displayed_summary_source"] = "user manuscript table; S2 Multitask baseline corrected to 25.07"
    comparison["test_source"] = "paired subject-level test accuracy, n=29; original CSV analysis"
    comparison["mean_difference_from_subject_data"] = comparison["mean"] - comparison["mean_from_subject_data"]
    comparison["SD_difference_from_subject_data"] = comparison["SD"] - comparison["SD_from_subject_data"]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / f"cross_session_labram{ext}" for ext in (".pdf", ".svg", ".png", "_preview.png")]
    outputs += [out / name for name in ("plot_values.csv", "figure_layout.json")]
    inputs = {args.table_input.resolve(), args.raw_statistics.resolve(), args.subject_data.resolve()}
    if inputs.intersection(p.resolve() for p in outputs):
        raise ValueError("An output would overwrite an input")
    points = None if args.no_subject_points else pd.read_csv(args.subject_data, dtype={"subject_id": str})
    colors = ({"Baseline": "#8c8c8c", "Rest-LoRA": "#e6862b", "Rest-Full": "#2e8b57"}
              if args.style == "classic" else None)
    layout = render_figure(comparison, out, colors=colors, style=args.style, subject_data=points,
                           figsize=(5.5, 3.45) if args.style == "reference" else (5.5, 3.15))
    if points is not None:
        layout["subject_point_source"] = str(args.subject_data.resolve())
        layout["subject_point_source_sha256"] = hashlib.sha256(args.subject_data.read_bytes()).hexdigest()
    layout["displayed_summary_source"] = str(args.table_input.resolve())
    layout["paired_statistics_source"] = str(args.raw_statistics.resolve())
    layout["table_input_sha256"] = hashlib.sha256(args.table_input.read_bytes()).hexdigest()
    layout["raw_statistics_sha256"] = hashlib.sha256(args.raw_statistics.read_bytes()).hexdigest()
    comparison.to_csv(out / "plot_values.csv", index=False, na_rep="NA", float_format="%.12g")
    (out / "figure_layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    print(comparison[key + ["mean", "SD", "n", "raw_p"]].to_string(index=False))
    print(f"\nManuscript-table version saved to {out}")


if __name__ == "__main__":
    main()
