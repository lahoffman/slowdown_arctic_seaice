#!/usr/bin/env python3
"""
09_gmt_coupling_check.py — does the onset-year coupling (Eq. coupling, App. A) matter for the GMT
slowdown problem of Labe & Barnes (2022) as much as for sea ice? (AIES §9; minutes, CPU-light)

For CESM2-LE yearly GMT and September SIE alike: test R² of the decadal trend anomaly regressed on
the state anomaly at onset t, for windows starting at t, t+1, t+2 (member-block splits), beside the
white-noise expectation 0.25 / 0 / 0. A large drop from offset 0 to 1 means the state's apparent
skill is mostly the coupling; a small drop means it is mostly persistence physics.

Outputs: results/coupling/gmt_vs_sie_offset_r2.nc, summary.md, FIGURES_DIR/diagnostics/gmt_coupling_check.png

Usage:
  python scripts/09_gmt_coupling_check.py [--window 10] [--end-year 2029]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl
from src.analysis.interannual import offset_r2
from src.data.cesm2le.slowdowns_gmt import load_gmt_yearly
from src.data.cesm2le.slowdowns_relative import group_mean_trends


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029)
    p.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def white_noise_r2(window: int, offset: int) -> float:
    if offset > 0:
        return 0.0
    return 3 * (window - 1) / (window * (window + 1))


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "coupling"; out_dir.mkdir(parents=True, exist_ok=True)
    onsets = np.arange(a.start_year, a.end_year + 1)

    gmt, gyears = load_gmt_yearly(str(paths.CESM2LE_TREF_DIR / "gmt"), start_year=1990, end_year=2100)
    gmt_anom = gmt - group_mean_trends(gmt, "group")
    sie, syears = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", np.arange(1990, 2101), demean="group")[0], np.arange(1990, 2101)
    sie_anom = sie - group_mean_trends(sie, "group")

    res = {}
    for name, series, anom, yrs in (("GMT", gmt, gmt_anom, gyears), ("SIE", sie, sie_anom, syears)):
        idx = np.searchsorted(yrs, onsets)
        res[name] = offset_r2(series, anom[:, idx], yrs, onsets, a.offsets, a.window)
        print(f"{name}: " + "  ".join(f"offset {o}: R² {np.nanmedian(v):.3f} ({np.nanmin(v):.3f}–{np.nanmax(v):.3f})" for o, v in res[name].items()))
    wn = {o: white_noise_r2(a.window, o) for o in a.offsets}

    ds = xr.Dataset({f"{n}_r2": (("offset", "split"), np.array([res[n][o] for o in a.offsets])) for n in res}
                    | {"white_noise_r2": ("offset", [wn[o] for o in a.offsets])},
                    coords={"offset": a.offsets, "split": np.arange(9)})
    ds.attrs.update(window=a.window, onsets=f"{a.start_year}-{a.end_year}")
    ds.to_netcdf(out_dir / "gmt_vs_sie_offset_r2.nc")
    lines = [f"# Onset-year coupling: GMT vs SIE (CESM2-LE, {a.window}-yr windows, onsets {a.start_year}–{a.end_year})\n",
             "Test R² of the decadal trend anomaly on the state anomaly at onset (median over 9 member-block splits).\n",
             "| window start | white noise | GMT | SIE |", "|---|---|---|---|"]
    for o in a.offsets:
        lines.append(f"| t+{o} | {wn[o]:.3f} | {np.nanmedian(res['GMT'][o]):.3f} | {np.nanmedian(res['SIE'][o]):.3f} |")
    g0, g1 = np.nanmedian(res["GMT"][0]), np.nanmedian(res["GMT"][1]); s0, s1 = np.nanmedian(res["SIE"][0]), np.nanmedian(res["SIE"][1])
    lines.append(f"\nShare of the onset-inclusive R² that disappears with the offset: GMT {(g0 - g1) / g0:.0%}, SIE {(s0 - s1) / s0:.0%}.\n"
                 "The coupling itself is identical for both (same window arithmetic); what differs is how much genuine "
                 "persistence each state has and how strong a proxy the input field is for it (OHC100→GMT vs Arctic SST→SIE).\n")
    md = "\n".join(lines) + "\n"; (out_dir / "summary.md").write_text(md); print("\n" + md)

    if not a.no_fig:
        from src.plotting import style as st
        from src.plotting.style import plt
        st.paper_rc()
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(a.offsets)); w = 0.36
        for i, (name, col) in enumerate((("GMT", st.ORANGE), ("SIE", st.BLUE))):
            med = [np.nanmedian(res[name][o]) for o in a.offsets]
            ax.bar(x + (i - 0.5) * w, med, width=w * 0.92, color=col, label=f"{name} state at onset → trend")
            for j, o in enumerate(a.offsets):
                ax.scatter(np.full(9, x[j] + (i - 0.5) * w), res[name][o], s=14, color=st.INK, alpha=0.6, zorder=3)
        ax.plot(x, [wn[o] for o in a.offsets], "k_", ms=28, mew=2.2, label="white noise (Eq. coupling)")
        ax.set_xticks(x); ax.set_xticklabels([f"window starts t+{o}" for o in a.offsets])
        ax.set_ylabel("test R² of the decadal trend anomaly"); ax.set_ylim(0, None)
        ax.legend(frameon=False, loc="upper right"); st.tidy(ax)
        ax.set_title("Onset-year coupling: GMT vs sea-ice slowdowns (CESM2-LE)", loc="left", weight="bold")
        out = paths.FIGURES_DIR / "diagnostics" / "gmt_coupling_check.png"
        fig.tight_layout(); fig.savefig(out, dpi=200); print(f"  figure → {out}")


if __name__ == "__main__":
    main()
