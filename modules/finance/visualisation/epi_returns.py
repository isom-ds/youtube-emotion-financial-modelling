import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
import datetime as dt

def epi_returns_plot(
    df: pd.DataFrame,
    *,
    filter_date: dt.date = dt.date(2025, 1, 13),
    event_windows=None,
    figsize=(20, 16),
    legend_ncol: int = 5,
    font_base: int = 18,
):
    if event_windows is None:
        event_windows = [
            ("2025-01-20", "2025-01-26"),
            ("2025-03-04", "2025-03-06"),
            ("2025-03-10", "2025-03-13"),
            ("2025-04-02", "2025-04-06"),
            ("2025-04-09", "2025-04-12"),
            ("2025-04-27", "2025-05-04"),
            ("2025-05-05", "2025-05-14"),
            ("2025-05-20", "2025-05-25"),
        ]

    left_series = [
        # Lowest priority: EI (thin + lighter + different dash patterns)
        ("ei_creator",   dict(ls=":", lw=2.0, label="Creator EI",   color="#34772E")),
        ("ei_community", dict(ls=":",  lw=2.0, label="Community EI", color="#344178")),

        # Mid priority: EPI signals (thicker + darker)
        ("epi",          dict(ls="--",  lw=2.0, label="EPI",              color="#AE2626")),
        ("epi_weighted", dict(ls="--", lw=2.0, label="Topic-Weighted EPI", color="#5B216D")),
    ]

    panels = [
        ("Bitcoin", "r_btc"),
        ("Gold",    "r_gold"),
        ("SPX",     "r_spx"),
    ]

    # Returns style: consistent across panels
    returns_style = dict(ls="-", lw=3.0, color="#252525", label="Asset Returns")

    # ---- data prep ----
    df1 = deepcopy(df)
    df1["date"] = pd.to_datetime(df1.index, errors="coerce")
    df1 = df1.dropna(subset=["date"])
    df1 = df1[df1["date"].dt.date >= filter_date].copy()
    df1 = df1.sort_values("date")

    # Ensure numeric (silent object dtypes can "plot" invisibly / oddly)
    for col in ["ei_creator", "ei_community", "epi", "epi_weighted", "r_btc", "r_gold", "r_spx"]:
        if col in df1.columns:
            df1[col] = pd.to_numeric(df1[col], errors="coerce")

    with plt.rc_context({
        "font.size": font_base,
        "axes.labelsize": font_base,
        "axes.titlesize": font_base + 2,
        "xtick.labelsize": font_base - 2,
        "ytick.labelsize": font_base - 2,
        "legend.fontsize": font_base - 2,
    }):
        fig, axs = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
        if len(panels) == 1:
            axs = [axs]
        right_axes = [ax.twinx() for ax in axs]

        # ✅ Transparent right axis, but drawn ABOVE left axis so returns are visible
        for ax_l, ax_r in zip(axs, right_axes):
            ax_r.patch.set_visible(False)
            ax_r.set_zorder(ax_l.get_zorder() + 1)

        def shade(ax_l):
            for s, e in event_windows:
                sdt, edt = pd.to_datetime(s), pd.to_datetime(e)
                ax_l.axvspan(sdt, edt, color="0.88", alpha=0.7, zorder=0)

        def polish(ax_l, ax_r, panel_title: str):
            ax_l.set_ylabel("EI / EPI", fontweight="bold")
            ax_r.set_ylabel(f"{panel_title} Returns", fontweight="bold")

            ax_l.grid(True, axis="y", alpha=0.25)
            ax_l.set_axisbelow(True)

            ax_l.spines["top"].set_visible(False)
            ax_r.spines["top"].set_visible(False)

            ax_l.set_title(panel_title, loc="left", pad=6)

        for (ax_l, ax_r), (title, r_col) in zip(zip(axs, right_axes), panels):
            shade(ax_l)

            # Left axis series
            for col, sty in left_series:
                ax_l.plot(df1["date"], df1[col], zorder=3, **sty)

            # Right axis returns
            ax_r.plot(df1["date"], df1[r_col], zorder=4, **returns_style)

            polish(ax_l, ax_r, title)

        # Weekly x ticks
        week_starts = df1["date"].dt.to_period("W").drop_duplicates().dt.start_time
        axs[-1].set_xticks(week_starts)
        axs[-1].set_xticklabels(week_starts.dt.strftime("%d-%b"), rotation=45, ha="right")

        # Legend from all axes
        handles, labels = [], []
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)
        for ax in right_axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h); labels.extend(l)

        seen, uniq_h, uniq_l = set(), [], []
        for h, l in zip(handles, labels):
            if l and (l not in seen):
                uniq_h.append(h); uniq_l.append(l); seen.add(l)

        fig.legend(
            uniq_h, uniq_l,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=legend_ncol,
            frameon=False,
        )

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show()
