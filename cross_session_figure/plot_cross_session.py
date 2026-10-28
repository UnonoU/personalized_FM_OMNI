import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/dataset4/yinuo/Rest_tuning"
OUT_DIR = os.path.join(HERE, "raw_data_version")

METHODS = ["Baseline", "Rest-LoRA", "Rest-Full"]
METHOD_COLORS = {"Baseline": "#BCC3CC", "Rest-LoRA": "#E49D87", "Rest-Full": "#92C3B4"}
SESSIONS = [1, 2, 3]
SESSION_TAGS = {1: "Calibration session", 2: "+1 week", 3: "+2 weeks"}
SESSION_XTICK = {s: f"Session {s}\n{SESSION_TAGS[s]}" for s in SESSIONS}

SETTINGS = {
    "Multitask": {
        "panel": "(a) Multitask",
        "sessions": {
            1: {
                "Baseline": f"{ROOT}/baseline/labram/multitask_COG/train0.1/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/Multitask/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/Multitask/train_0.1/cogbci/linear/transfer_summary_linear.csv",
            },
            2: {
                "Baseline": f"{ROOT}/result/LaBraM/baseline_CrossSession/Multitask_ses2_new/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/classification/COG_Multitask_CrossSession/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/classification/COG_Multitask_CrossSession/train_0.1/cogbci/linear/transfer_summary_linear.csv",
            },
            3: {
                "Baseline": f"{ROOT}/result/LaBraM/baseline_CrossSession/Multitask_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/classification/COG_Multitask_CrossSession_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/classification/COG_Multitask_CrossSession_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
            },
        },
    },
    "NBack": {
        "panel": "(b) NBack",
        "sessions": {
            1: {
                "Baseline": f"{ROOT}/baseline/labram/nback/train0.1/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/classification/nback/train_0.1_b8/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/classification/nback/train_0.1_b8/cogbci/linear/transfer_summary_linear.csv",
            },
            2: {
                "Baseline": f"{ROOT}/result/LaBraM/baseline_CrossSession/classification/Nback_ses2/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/classification/nback_CrossSession/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/classification/nback_CrossSession/train_0.1/cogbci/linear/transfer_summary_linear.csv",
            },
            3: {
                "Baseline": f"{ROOT}/result/LaBraM/baseline_CrossSession/classification/Nback_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-LoRA": f"{ROOT}/result/LaBraM/lora_mask_0.7/wo_label_channel_0418/classification/nback_CrossSession_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
                "Rest-Full": f"{ROOT}/result/LaBraM/full_mask_0.3/wo_label_channel_0418/classification/nback_CrossSession_ses3/train_0.1/cogbci/linear/transfer_summary_linear.csv",
            },
        },
    },
}

ACC_COL = "test_acc"
SUBJ_COL = "subject_id"


def load_subject_acc(path):
    """Return a Series of test accuracy (%) indexed by subject id.

    Drops summary rows (non-integer subject ids), converts fractions to
    percentages exactly once, and averages multiple rows of the same subject.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if SUBJ_COL not in df.columns or ACC_COL not in df.columns:
        raise ValueError(f"{path}: missing {SUBJ_COL}/{ACC_COL} columns")
    subj = pd.to_numeric(df[SUBJ_COL], errors="coerce")
    n_raw = len(df)
    n_summary = int(subj.isna().sum())
    df = df.loc[subj.notna()].copy()
    df[SUBJ_COL] = subj.loc[subj.notna()].astype(int)
    acc = pd.to_numeric(df[ACC_COL], errors="raise")
    if acc.max() <= 1.5:  # fractions -> percent, applied exactly once
        acc = acc * 100.0
    df["_acc"] = acc
    dup_mask = df.duplicated(SUBJ_COL, keep=False)
    n_duplicated = int(df.loc[dup_mask, SUBJ_COL].nunique())
    ser = df.groupby(SUBJ_COL)["_acc"].mean()
    return ser, {
        "path": path,
        "n_rows": n_raw,
        "n_summary_rows_dropped": n_summary,
        "n_subjects_unique": int(ser.size),
        "n_subjects_with_duplicate_rows": n_duplicated,
        "acc_min": float(ser.min()),
        "acc_max": float(ser.max()),
    }


def holm_adjust(pvals):
    """Holm-adjusted p values (same family as statsmodels multipletests 'holm')."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = (m - np.arange(m)) * ranked
    adj = np.maximum.accumulate(adj)
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty(m, dtype=float)
    out[order] = adj
    return out


def stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def analyze(settings):
    data = {}       # setting -> session -> method -> Series
    audit = []      # per-file audit rows
    stats_rows = []
    comp_rows = []
    for setting in settings:
        sconf = SETTINGS[setting]
        for ses in SESSIONS:
            aligned = {}
            for method in METHODS:
                ser, info = load_subject_acc(sconf["sessions"][ses][method])
                data.setdefault(setting, {}).setdefault(ses, {})[method] = ser
                audit.append({"setting": setting, "session": ses, "method": method, **info})
                aligned[method] = ser
            common = set.intersection(*[set(s.index) for s in aligned.values()])
            union = set.union(*[set(s.index) for s in aligned.values()])
            missing = {m: sorted(union - set(aligned[m].index)) for m in METHODS if union - set(aligned[m].index)}
            for method in METHODS:
                ser_all = aligned[method]
                ser = ser_all.loc[sorted(common)]
                stats_rows.append({
                    "setting": setting,
                    "session": ses,
                    "method": method,
                    "n_subjects_paired": len(common),
                    "n_subjects_file": int(ser_all.size),
                    "missing_subjects": ";".join(map(str, missing.get(method, []))),
                    "mean": ser.mean(),
                    "sd": ser.std(ddof=1),
                })
            base = aligned["Baseline"].loc[sorted(common)]
            for method in ["Rest-LoRA", "Rest-Full"]:
                other = aligned[method].loc[sorted(common)]
                t, p = stats.ttest_rel(other, base)
                comp_rows.append({
                    "setting": setting,
                    "session": ses,
                    "comparison": f"Baseline vs {method}",
                    "n_pairs": len(common),
                    "mean_gain": (other - base).mean(),
                    "paired_t": t,
                    "df": len(common) - 1,
                    "raw_p": p,
                })
    stats_df = pd.DataFrame(stats_rows)
    comp_df = pd.DataFrame(comp_rows)
    base_mask = stats_df["method"] == "Baseline"
    for setting in settings:
        sm = stats_df["setting"] == setting
        for ses in SESSIONS:
            b = stats_df.loc[sm & (stats_df["session"] == ses) & base_mask, "mean"]
            if len(b):
                stats_df.loc[sm & (stats_df["session"] == ses), "gain_over_baseline"] = (
                    stats_df.loc[sm & (stats_df["session"] == ses), "mean"] - float(b.iloc[0])
                )
    n_comp = len(comp_df)
    holm_all_col = f"holm_p_all_{n_comp}"
    comp_df["holm_p_within_setting"] = np.nan
    comp_df[holm_all_col] = np.nan
    for setting in settings:
        mask = comp_df["setting"] == setting
        comp_df.loc[mask, "holm_p_within_setting"] = holm_adjust(comp_df.loc[mask, "raw_p"].values)
    comp_df[holm_all_col] = holm_adjust(comp_df["raw_p"].values)
    comp_df["significant_nominal"] = comp_df["raw_p"] < 0.05
    return data, pd.DataFrame(audit), stats_df, comp_df


def make_figure(data, comp_df, settings, out_basename):
    """Use the same reference style as the manuscript figure, with raw statistics."""
    from plot_style import render_figure

    rows = []
    point_rows = []
    all_holm = next(c for c in comp_df.columns if c.startswith("holm_p_all_"))
    for setting in settings:
        for ses in SESSIONS:
            methods = data[setting][ses]
            common = sorted(set.intersection(*(set(methods[m].index) for m in METHODS)))
            for method in METHODS:
                ser = methods[method].loc[common]
                row = {
                    "setting": setting, "session": ses, "method": method,
                    "n": len(common), "mean": float(ser.mean()),
                    "SD": float(ser.std(ddof=1)), "raw_p": np.nan,
                    "paired_t": np.nan, "holm_adjusted_p": np.nan,
                    "holm_adjusted_p_all": np.nan,
                }
                if method != "Baseline":
                    match = comp_df.loc[(comp_df["setting"] == setting) &
                                        (comp_df["session"] == ses) &
                                        (comp_df["comparison"] == f"Baseline vs {method}")]
                    if len(match) != 1:
                        raise ValueError(f"Expected one paired comparison: {setting}, {ses}, {method}")
                    test = match.iloc[0]
                    row.update(raw_p=float(test["raw_p"]), paired_t=float(test["paired_t"]),
                               holm_adjusted_p=float(test["holm_p_within_setting"]),
                               holm_adjusted_p_all=float(test[all_holm]))
                rows.append(row)
                for subject_id, accuracy in ser.items():
                    point_rows.append({"setting": setting, "session": ses,
                                       "method": method, "subject_id": str(subject_id),
                                       "accuracy_pct": float(accuracy)})
    points = pd.DataFrame(point_rows)
    layout = render_figure(
        pd.DataFrame(rows), Path(OUT_DIR), basename=out_basename,
        settings=tuple(settings), style="reference", colors=METHOD_COLORS,
        subject_data=points, figsize=(5.5 if len(settings) == 2 else 3.6, 3.45),
    )
    layout["summary_source"] = "recalculated from original subject-level CSV files"
    layout["statistics_unchanged"] = "mean, sample SD and paired tests use the original analysis functions"
    suffix = out_basename.removeprefix("cross_session_labram")
    points.to_csv(Path(OUT_DIR) / f"cross_session_subject_accuracy{suffix}.csv", index=False)
    (Path(OUT_DIR) / f"figure_layout{suffix}.json").write_text(json.dumps(layout, indent=2) + "\n")
    return layout


def write_outputs(audit_df, stats_df, comp_df, suffix):
    stats_out = stats_df.copy()
    for c in ["mean", "sd", "gain_over_baseline"]:
        stats_out[c] = stats_out[c].round(4)
    stats_out.to_csv(os.path.join(OUT_DIR, f"cross_session_statistics{suffix}.csv"), index=False)
    comp_out = comp_df.copy()
    for c in comp_out.columns:
        if c.startswith(("mean_gain", "paired_t", "raw_p", "holm_p")):
            comp_out[c] = comp_out[c].astype(float).round(6)
    comp_out.to_csv(os.path.join(OUT_DIR, f"cross_session_comparisons{suffix}.csv"), index=False)
    return stats_out, comp_out


def main(argv):
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Recalculate original subject-level CSVs and plot the reference style.")
    parser.add_argument("settings", nargs="*", help="Optional setting names: Multitask NBack")
    parser.add_argument("--data-root", default=ROOT, help="Rest_tuning root on L40 or its local mount")
    parser.add_argument("--output-dir", default=OUT_DIR, help="Separate output directory for raw recalculation")
    args = parser.parse_args(argv[1:])
    unknown = set(args.settings) - set(SETTINGS)
    if unknown:
        parser.error(f"Unknown settings: {sorted(unknown)}")
    selected = list(dict.fromkeys(args.settings))
    if not selected:
        selected = list(SETTINGS)
    data_root = os.path.abspath(os.path.expanduser(args.data_root))
    if data_root == ROOT and not os.path.isdir(data_root):
        mounted = ROOT.replace("/mnt/dataset4/", "/Volumes/dataset4/", 1)
        if os.path.isdir(mounted):
            data_root = mounted
    if not os.path.isdir(data_root):
        parser.error(f"Data root not found: {data_root}")
    for conf in SETTINGS.values():
        for methods in conf["sessions"].values():
            for method, path in list(methods.items()):
                if path.startswith(ROOT + "/"):
                    methods[method] = data_root + path[len(ROOT):]
    OUT_DIR = os.path.abspath(os.path.expanduser(args.output_dir))
    source_paths = {Path(path).resolve() for conf in SETTINGS.values()
                    for methods in conf["sessions"].values() for path in methods.values()}
    protected_names = ["cross_session_statistics.csv", "manuscript_table_values.csv", "paired_subject_accuracy.csv"]
    source_paths.update(Path(HERE, name).resolve() for name in protected_names)
    suffix = "" if len(selected) == len(SETTINGS) else "_" + "_".join(s.lower() for s in selected)
    generated_names = [f"cross_session_statistics{suffix}.csv", f"cross_session_comparisons{suffix}.csv",
                       f"cross_session_data_check{suffix}.md", f"cross_session_subject_accuracy{suffix}.csv",
                       "figure_layout" + suffix + ".json"]
    generated_names += ["cross_session_labram" + suffix + ext for ext in (".pdf", ".svg", ".png", "_preview.png")]
    if source_paths.intersection(Path(OUT_DIR, name).resolve() for name in generated_names):
        parser.error("Output would overwrite an input used by the manuscript-table workflow; choose another --output-dir")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_basename = "cross_session_labram" + suffix
    data, audit_df, stats_df, comp_df = analyze(selected)
    stats_out, comp_out = write_outputs(audit_df, stats_df, comp_df, suffix)
    make_figure(data, comp_df, selected, out_basename)

    pd.set_option("display.width", 200)
    print("=== File audit ===")
    print(audit_df.to_string(index=False))
    print("\n=== Summary statistics (%) ===")
    print(stats_out.to_string(index=False))
    print("\n=== Paired comparisons ===")
    print(comp_out.to_string(index=False))

    def md_table(df):
        try:
            return df.to_markdown(index=False)
        except ImportError:
            return "```\n" + df.to_string(index=False) + "\n```"

    md = ["# Cross-session linear-probing data check (LaBraM)", ""]
    md.append("Source: subject-level `test_acc` from each `transfer_summary_linear.csv`;")
    md.append("rows with non-integer subject ids (mean/std summaries) dropped;")
    md.append("fractions converted to percent exactly once; duplicate subject rows averaged (none expected).")
    md.append("")
    md.append("## File audit")
    md.append(md_table(audit_df))
    md.append("")
    md.append("## Summary statistics")
    md.append(md_table(stats_out))
    md.append("")
    md.append("## Paired t-tests (two-sided)")
    md.append(md_table(comp_out))
    md.append("")
    md.append("Stars in the figure denote *nominal* (uncorrected) paired p values;")
    md.append("Holm-adjusted p values are reported in `cross_session_comparisons.csv` only.")
    md.append("")
    md.append("## LaTeX figure caption")
    md.append("```latex")
    md.append(make_caption(selected, out_basename).strip())
    md.append("```")
    with open(os.path.join(OUT_DIR, f"cross_session_data_check{suffix}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\nOutputs saved to: {OUT_DIR}")


def make_caption(settings, out_basename):
    body = (
        "Cross-session linear-probing accuracy of LaBraM on COGBCI {tasks} classification. "
        "The backbone is calibrated once on Session-1 resting EEG (Rest-LoRA, Rest-Full) or used "
        "without calibration (Baseline), then frozen; no further resting calibration is performed "
        "in Sessions 2 (+1 week) and 3 (+2 weeks). For each session, a linear head is trained per "
        "subject with 10\\% of that session's task trials for training and a separate approximately "
        "10\\% validation split for checkpoint selection, then evaluated on the held-out test split. "
        "Bars show subject-level mean accuracy ($n=29$) with error bars indicating the across-subject "
        "sample SD; dots show individual test accuracies. Brackets mark nominal significance of two-sided paired t-tests against Baseline "
        "(*p<0.05, **p<0.01, ***p<0.001; uncorrected for multiple comparisons)."
    )
    if len(settings) == 2:
        tasks = "multitask (a) and NBack (b)"
    else:
        tasks = f"{settings[0].lower()}"
    return CAPTION_TEMPLATE.format(pdf=out_basename, body=body.replace("{tasks}", tasks))


CAPTION_TEMPLATE = r"""
\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{{pdf}.pdf}}
\caption{{{body}}}
\label{{fig:cross_session}}
\end{{figure}}
"""

if __name__ == "__main__":
    sys.exit(main(sys.argv))
