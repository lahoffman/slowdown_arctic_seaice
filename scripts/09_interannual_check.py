#!/usr/bin/env python3
"""
09_interannual_check.py — two sanity checks behind the decadal result (step 8.6). Minutes, no GPU.

1. Does CESM2-LE have the year-to-year Pacific → September SIE link (Ding/Baxter bridge),
   and how nonstationary is it (per-member and 30-yr running correlations; year-ahead
   regression of SIE(t) on SIE(t−1) with and without concurrent JJA indices)?
2. How much of "SIE(t) predicts the decadal trend" is the onset year sitting inside the
   trend window? R² for windows starting at t, t+1, t+2.

Outputs: results/interannual/interannual_summary.md, FIGURES_DIR/diagnostics/interannual_check.png
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl, interannual as ia, residual as rs
from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.data.cesm2le.slowdowns_relative import group_mean_trends


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2030)
    p.add_argument("--no-map", action="store_true"); p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "interannual"; out_dir.mkdir(parents=True, exist_ok=True)
    metrics = paths.CESM2LE_AICE_DIR / "metrics"
    sie_full, yrs_full = load_sie_monthly_files(str(metrics), "SEP", variable="sie", start_year=1990, end_year=2100)
    years = np.arange(a.start_year, a.end_year + 1)
    _, sie_anom = bl.load_sie_anomaly(metrics, years, demean="group")
    idx = bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, a.start_year, a.end_year)
    print(f"09  —  interannual check  {a.start_year}–{a.end_year}, indices {sorted(idx)}")

    member_r = {k: ia.member_correlations(sie_anom, v) for k, v in idx.items()}
    running = {k: ia.running_correlations(sie_anom, v, years, 30)[0] for k, v in idx.items()}
    run_years = ia.running_correlations(sie_anom, idx["nino34"], years, 30)[1]
    lag_r2 = ia.lagged_regression_r2(sie_anom, idx, years)
    # trend windows starting at t, t+1, t+2 on the group-demeaned SIE
    sie_dem = sie_full - group_mean_trends(sie_full, "group")
    off_r2 = ia.offset_r2(sie_dem, sie_anom, yrs_full, years, offsets=(0, 1, 2))

    med = lambda v: float(np.nanmedian(v))
    lines = ["# Interannual sanity checks (CESM2-LE, onsets/years %d–%d)\n" % (a.start_year, a.end_year),
             "## 1. Year-to-year Pacific → September SIE link\n",
             "| index | per-member corr median | 10–90 % across members | members with r > 0 | 30-yr running corr range (median member) |",
             "|---|---|---|---|---|"]
    for k, r in member_r.items():
        rr = np.median(running[k], 0)
        lines.append(f"| {k} | {np.median(r):+.3f} | {np.percentile(r, 10):+.2f} … {np.percentile(r, 90):+.2f} | "
                     f"{int((r > 0).sum())}/100 | {rr.min():+.2f} … {rr.max():+.2f} |")
    lines += ["", "Year-ahead regression, test R² of SIE(t) (median over 9 member-block splits):", ""]
    for k, v in lag_r2.items():
        lines.append(f"- {k}: **{med(v):.3f}**" + ("" if k == "persistence" else f" (Δ {med(v) - med(lag_r2['persistence']):+.3f})"))
    lines += ["", "## 2. Onset year inside vs outside the trend window\n",
              "Test R² of the 10-yr group-relative trend anomaly regressed on SIE anomaly(t):", ""]
    for o, v in off_r2.items():
        lines.append(f"- window t+{o} … t+{o + 9}: **{med(v):.3f}** (range {np.nanmin(v):.3f}–{np.nanmax(v):.3f})")
    lines.append("\nThe drop from offset 0 to offset 1 is the share of the ice-state skill that is the onset value "
                 "sitting at the start of the fitted window (arithmetic), the remainder is memory.")
    md = "\n".join(lines) + "\n"
    (out_dir / "interannual_summary.md").write_text(md); print("\n" + md)

    if a.no_map and a.no_fig:
        return
    corr_map = None
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    if not a.no_map:
        from src.cnn.splits import load_jja_sst_demeaned
        with nc.Dataset(paths.LANDMASK_FILE) as d:
            land = np.array(d["landmask"][:]) == 1
        print("loading concurrent JJA SST ...")
        sst, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=a.start_year, end_year=a.end_year, demean="group")
        corr_map = np.where(land, np.nan, rs.correlation_map(sie_anom, sst)); del sst
    if not a.no_fig:
        from src.plotting import interannual as plot, style as st
        st.paper_rc()
        plot.plot_interannual(member_r, running, run_years, lag_r2, off_r2,
                              corr_map if corr_map is not None else np.full((lat.size, lon.size), np.nan), lat, lon,
                              paths.FIGURES_DIR / "diagnostics" / "interannual_check.png")


if __name__ == "__main__":
    main()
