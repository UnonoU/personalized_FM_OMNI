from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
import numpy as np
import pandas as pd


METHODS = ("Baseline", "Rest-LoRA", "Rest-Full")
DEFAULT_COLORS = {
    "Baseline": "#999999",
    "Rest-LoRA": "#E69F00",
    "Rest-Full": "#009E73",
}
REFERENCE_COLORS = {
    # Invert the white-background alpha=.85 blend to match sampled JPEG fills:
    # displayed approximately #C6CCD4 / #E8AC99 / #A2CCBF.
    "Baseline": "#BCC3CC",
    "Rest-LoRA": "#E49D87",
    "Rest-Full": "#92C3B4",
}


def _stars(p_value: float) -> str:
    """Strict, two-sided nominal significance thresholds."""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _verified_panel_rows(statistics: pd.DataFrame, setting: str) -> pd.DataFrame:
    required = {"setting", "session", "method", "n", "mean", "SD", "raw_p"}
    missing_columns = sorted(required.difference(statistics.columns))
    if missing_columns:
        raise ValueError(f"Statistics are missing required columns: {missing_columns}")
    rows = statistics.loc[statistics["setting"].eq(setting)].copy()
    if rows.empty:
        raise ValueError(f"No verified statistics are available for {setting!r}.")
    rows["session"] = pd.to_numeric(rows["session"], errors="raise")
    if not rows["session"].isin((1, 2, 3)).all():
        raise ValueError(f"{setting}: session must be one of 1, 2, 3.")
    if rows.duplicated(["session", "method"]).any():
        duplicates = rows.loc[
            rows.duplicated(["session", "method"], keep=False), ["session", "method"]
        ].to_dict("records")
        raise ValueError(f"{setting}: duplicate statistics conditions: {duplicates}")
    expected = {(session, method) for session in (1, 2, 3) for method in METHODS}
    actual = set(zip(rows["session"].astype(int), rows["method"]))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            f"{setting}: a complete panel requires all nine verified conditions; "
            f"missing={missing}, unexpected={unexpected}. No values will be imputed."
        )
    for column in ("n", "mean", "SD"):
        rows[column] = pd.to_numeric(rows[column], errors="raise")
        if not np.isfinite(rows[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{setting}: {column} contains a missing or non-finite value.")
    if ((rows["n"] < 2) | (rows["n"] % 1 != 0)).any():
        raise ValueError(f"{setting}: n must be an integer of at least two subjects.")
    if ((rows["mean"] < 0) | (rows["mean"] > 100)).any():
        raise ValueError(f"{setting}: mean must already be an accuracy percentage.")
    if (rows["SD"] < 0).any():
        raise ValueError(f"{setting}: SD cannot be negative.")
    for session, group in rows.groupby("session", sort=True):
        if group["n"].nunique() != 1:
            raise ValueError(
                f"{setting}, session {session}: the three methods must use the same "
                "aligned subject cohort."
            )
    rest = rows["method"].ne("Baseline")
    rows.loc[rest, "raw_p"] = pd.to_numeric(rows.loc[rest, "raw_p"], errors="raise")
    p_values = rows.loc[rest, "raw_p"].to_numpy(dtype=float)
    if not np.isfinite(p_values).all() or ((p_values < 0) | (p_values > 1)).any():
        raise ValueError(f"{setting}: all six rest-versus-baseline raw p-values are required.")
    return rows.set_index(["session", "method"]).sort_index()


def _bbox_list(bbox) -> list[float]:
    return [round(float(value), 3) for value in bbox.extents]


def _check_layout(fig, legend, axes, bracket_texts) -> dict:
    """Check rendered pixel bounds before writing the camera-ready artifacts."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_box = fig.bbox
    legend_box = legend.get_window_extent(renderer)
    legend_children = set(legend.findobj())
    outside = []
    legend_overlaps = []
    visible_text_count = 0
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width == 0 or bbox.height == 0:
            continue
        visible_text_count += 1
        # A one-pixel tolerance avoids floating-point noise at a canvas edge.
        if (
            bbox.x0 < canvas_box.x0 - 1
            or bbox.y0 < canvas_box.y0 - 1
            or bbox.x1 > canvas_box.x1 + 1
            or bbox.y1 > canvas_box.y1 + 1
        ):
            outside.append({"text": artist.get_text(), "bbox_px": _bbox_list(bbox)})
        if artist not in legend_children and legend_box.overlaps(bbox):
            legend_overlaps.append(artist.get_text())
    bracket_overlaps = []
    for index, (ax, text) in enumerate(bracket_texts):
        first = text.get_window_extent(renderer)
        for second_ax, second_text in bracket_texts[index + 1 :]:
            if ax is second_ax and first.overlaps(second_text.get_window_extent(renderer)):
                bracket_overlaps.append([text.get_text(), second_text.get_text()])
    result = {
        "canvas_px": [round(float(canvas_box.width), 3), round(float(canvas_box.height), 3)],
        "legend_bbox_px": _bbox_list(legend_box),
        "axes_bbox_px": [_bbox_list(ax.get_window_extent(renderer)) for ax in axes],
        "visible_text_count": visible_text_count,
        "cropped_text": outside,
        "legend_text_overlaps": legend_overlaps,
        "significance_text_overlaps": bracket_overlaps,
    }
    if outside or legend_overlaps or bracket_overlaps:
        raise ValueError(f"Figure layout check failed: {result}")
    return result


def render_figure(
    statistics: pd.DataFrame,
    output_dir: Path,
    basename: str = "cross_session_labram",
    settings: Sequence[str] = ("Multitask", "NBack"),
    figsize: tuple[float, float] = (5.5, 3.15),
    colors: Mapping[str, str] | None = None,
    style: str = "classic",
    subject_data: pd.DataFrame | None = None,
) -> dict:
    """Render complete, verified mean/SD panels and return layout metadata.

    ``statistics`` must have exactly one row per requested setting, session
    (integer 1--3), and method (Baseline, Rest-LoRA, Rest-Full).  Required
    columns are ``setting, session, method, n, mean, SD, raw_p``; mean and SD
    must already be in percentage points.  ``raw_p`` is the two-sided paired
    t-test p-value against baseline, and may be NaN for baseline rows only.

    Missing conditions, missing p-values, unequal within-session cohort sizes,
    duplicate conditions, and non-finite descriptive statistics raise a
    ValueError.  No missing value is created or filled here.  Holm-adjusted
    p-values, if present, are deliberately unused for figure annotations.

    Writes vector PDF/SVG, a 600-dpi PNG, and a 150-dpi preview PNG.  The
    metadata includes output paths, shared y-limits, bracket geometry, and
    actual rendered text/legend bounds checked before saving.
    """
    settings = tuple(settings)
    if not settings or len(set(settings)) != len(settings):
        raise ValueError("settings must contain at least one unique setting name.")
    if len(settings) > 2:
        raise ValueError("This final-size layout supports one or two panels.")
    if len(figsize) != 2 or any(float(size) <= 0 for size in figsize):
        raise ValueError("figsize must contain two positive inch dimensions.")
    if not basename or Path(basename).name != basename:
        raise ValueError("basename must be a filename stem without directory components.")
    if style not in {"classic", "reference"}:
        raise ValueError("style must be classic or reference")
    reference = style == "reference"
    palette = dict(REFERENCE_COLORS if reference else DEFAULT_COLORS)
    if colors is not None:
        unknown = set(colors).difference(METHODS)
        if unknown:
            raise ValueError(f"Unknown color method keys: {sorted(unknown)}")
        palette.update(colors)
    for color in palette.values():
        mpl.colors.to_rgba(color)
    panels = {setting: _verified_panel_rows(statistics, setting) for setting in settings}
    point_groups = {}
    if subject_data is not None:
        point_keys = ["setting", "session", "method"]
        required = set(point_keys + ["subject_id", "accuracy_pct"])
        if not required.issubset(subject_data.columns):
            raise ValueError("Subject points require setting/session/method/subject_id/accuracy_pct")
        points = subject_data.loc[subject_data.setting.isin(settings)].copy()
        if points.duplicated(point_keys + ["subject_id"]).any():
            raise ValueError("Each displayed subject may have only one point per condition")
        if not np.isfinite(points.accuracy_pct).all() or not points.accuracy_pct.between(0, 100).all():
            raise ValueError("Subject point accuracies must be finite percentages")
        for setting, panel in panels.items():
            for session in (1, 2, 3):
                ids = []
                for method in METHODS:
                    group = points.loc[(points.setting == setting) & (points.session == session) &
                                       (points.method == method)].sort_values("subject_id")
                    if len(group) != int(panel.loc[(session, method), "n"]):
                        raise ValueError(f"Subject point count does not match n: {setting}, {session}, {method}")
                    ids.append(set(group.subject_id))
                    point_groups[(setting, session, method)] = group.accuracy_pct.to_numpy(float)
                if not ids[0] == ids[1] == ids[2]:
                    raise ValueError("Subject point IDs must be paired across all three methods")
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    max_error_top = max(float((rows["mean"] + rows["SD"]).max()) for rows in panels.values())
    max_point = max((float(values.max()) for values in point_groups.values()), default=0.0)
    scale = max(max_error_top, max_point, 1.0)
    bar_width = 0.225
    offsets = {method: (index - 1) * bar_width for index, method in enumerate(METHODS)}
    bracket_records = []
    for setting, rows in panels.items():
        for session in (1, 2, 3):
            previous_annotation_top = None
            for method in METHODS[1:]:
                row = rows.loc[(session, method)]
                p_value = float(row["raw_p"])
                stars = _stars(p_value)
                if not stars:
                    continue
                span_methods = METHODS[: METHODS.index(method) + 1]
                span_error_top = max(
                    float(rows.loc[(session, other), "mean"] + rows.loc[(session, other), "SD"])
                    for other in span_methods
                )
                span_point_top = max((float(point_groups[(setting, session, other)].max())
                                      for other in span_methods if (setting, session, other) in point_groups),
                                     default=0.0)
                bracket_bottom = max(span_error_top, span_point_top) + (0.035 if reference else 0.050) * scale
                if previous_annotation_top is not None:
                    # Leave enough room for the first bracket's star glyphs.
                    bracket_bottom = max(bracket_bottom, previous_annotation_top + (0.090 if reference else 0.120) * scale)
                bracket_top = bracket_bottom + 0.020 * scale
                text_y = bracket_top + 0.006 * scale
                previous_annotation_top = text_y
                bracket_records.append(
                    {
                        "setting": setting,
                        "session": session,
                        "method": method,
                        "raw_p": p_value,
                        "stars": stars,
                        "x1": session - 1 + offsets["Baseline"],
                        "x2": session - 1 + offsets[method],
                        "span_error_top": span_error_top,
                        "span_point_top": span_point_top,
                        "bottom": bracket_bottom,
                        "top": bracket_top,
                        "text_y": text_y,
                    }
                )
    annotation_top = max((record["text_y"] for record in bracket_records), default=0.0)
    y_max = max(max(max_error_top, max_point) + 0.09 * scale, annotation_top + 0.11 * scale)
    rc = {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
    files = {
        "pdf": output_dir / f"{basename}.pdf",
        "svg": output_dir / f"{basename}.svg",
        "png": output_dir / f"{basename}.png",
        "preview": output_dir / f"{basename}_preview.png",
    }
    with mpl.rc_context(rc):
        fig, axes_array = plt.subplots(
            1, len(settings), figsize=figsize, sharey=True, squeeze=False, dpi=150
        )
        axes = list(axes_array[0])
        fig.subplots_adjust(left=0.165 if len(settings) == 1 else 0.095, right=0.990,
                            bottom=0.245 if reference else 0.275,
                            top=0.815 if reference else 0.790, wspace=0.130)
        bracket_texts = []
        rng = np.random.default_rng(42)
        try:
            for panel_index, (ax, setting) in enumerate(zip(axes, settings)):
                rows = panels[setting]
                x_positions = np.arange(3, dtype=float)
                for method in METHODS:
                    ordered = rows.xs(method, level="method").reindex((1, 2, 3))
                    ax.bar(
                        x_positions + offsets[method],
                        ordered["mean"].to_numpy(dtype=float),
                        width=bar_width * 0.93,
                        yerr=ordered["SD"].to_numpy(dtype=float),
                        color=mpl.colors.to_rgba(palette[method], 0.85) if reference else palette[method],
                        edgecolor="#53595B" if reference else "white",
                        linewidth=0.65 if reference else 0.35,
                        capsize=2,
                        error_kw={"elinewidth": 0.75, "capthick": 0.75, "ecolor": "#303030", "zorder": 5},
                        zorder=3,
                    )
                    point_colors = {"Baseline": "#485A65", "Rest-LoRA": "#A96C59", "Rest-Full": "#397B69"}
                    for session in (1, 2, 3):
                        values = point_groups.get((setting, session, method))
                        if values is not None:
                            jitter = rng.uniform(-0.040, 0.040, len(values))
                            ax.scatter(session - 1 + offsets[method] + jitter, values,
                                       s=3.0, color=point_colors[method], alpha=0.30,
                                       linewidths=0, zorder=4, rasterized=False)
                ax.set_xlim(-0.55, 2.55)
                ax.set_ylim(0, y_max)
                ax.set_xticks(x_positions, ("Session 1", "Session 2", "Session 3"))
                ax.tick_params(axis="x", length=0, pad=5)
                ax.tick_params(axis="y", length=3, width=0.65)
                tick_candidates = mpl.ticker.MaxNLocator(nbins=5, min_n_ticks=4).tick_values(0, y_max)
                ax.set_yticks(tick_candidates[(tick_candidates >= 0) & (tick_candidates <= y_max)])
                ax.grid(axis="y", color="#EEEEEE" if reference else "#E7E7E7", linewidth=0.55, zorder=0)
                ax.set_axisbelow(True)
                ax.spines[["top", "right"]].set_visible(False)
                ax.spines[["bottom", "left"]].set_color("#555555")
                ax.set_title(f"({chr(97 + panel_index)}) {setting}", loc="left", pad=9, weight="normal")
                for x_pos, label in zip(x_positions, ("Calibration\nsession", "+1 week", "+2 weeks")):
                    ax.text(
                        x_pos, -0.205, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7.5, color="#404040", clip_on=False,
                    )
                for record in bracket_records:
                    if record["setting"] != setting:
                        continue
                    ax.plot(
                        [record["x1"], record["x1"], record["x2"], record["x2"]],
                        [record["bottom"], record["top"], record["top"], record["bottom"]],
                        color="#303030", linewidth=0.65, solid_capstyle="butt", zorder=5,
                    )
                    text = ax.text(
                        (record["x1"] + record["x2"]) / 2,
                        record["text_y"], record["stars"], ha="center", va="bottom",
                        fontsize=9, color="#202020", zorder=6,
                    )
                    bracket_texts.append((ax, text))
            axes[0].set_ylabel("Linear probing accuracy (%)", labelpad=7)
            legend = fig.legend(
                handles=[Patch(facecolor=mpl.colors.to_rgba(palette[method], 0.85) if reference else palette[method],
                               edgecolor="#53595B" if reference else "none", linewidth=0.6, label=method)
                         for method in METHODS],
                loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3,
                frameon=False, handlelength=1.35, handleheight=0.85,
                columnspacing=1.7, handletextpad=0.5, borderaxespad=0.25,
            )
            layout = _check_layout(fig, legend, axes, bracket_texts)
            # Fixed-inch output preserves exactly the reviewed paper insertion size.
            fig.savefig(files["pdf"], format="pdf")
            fig.savefig(files["svg"], format="svg")
            fig.savefig(files["png"], format="png", dpi=600)
            fig.savefig(files["preview"], format="png", dpi=150)
        finally:
            plt.close(fig)
    return {
        "files": {kind: str(path) for kind, path in files.items()},
        "settings": list(settings),
        "figsize_inches": [float(value) for value in figsize],
        "shared_ylim": [0.0, y_max],
        "colors": palette,
        "style": style,
        "bar_alpha": 0.85 if reference else 1.0,
        "reference_sampled_fills": {"Baseline": "#C6CCD4", "Rest-LoRA": "#E8AC99", "Rest-Full": "#A2CCBF"}
                                   if reference else None,
        "subject_points": int(sum(len(values) for values in point_groups.values())),
        "jitter_seed": 42 if point_groups else None,
        "significance": "two-sided paired t-tests; unadjusted nominal p-values",
        "brackets": bracket_records,
        "layout_check": layout,
    }
