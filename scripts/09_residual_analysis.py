#!/usr/bin/env python3
"""
09_residual_analysis.py — does SST explain the part of the trend the ice state does not? (step 8.1)

Continuous target: the 10-yr September SIE trend anomaly (relative to the forcing
group), no threshold. Stage 1 fits trend ~ SIE anomaly at onset; stage 2 fits the
residual on the SST indices, either at onset or averaged over the trend decade
(concurrent). Also maps the correlation of the residual with JJA SST at onset and
with the decade-mean JJA SST. Baselines only — minutes, no GPU.

Outputs: results/residual/residual_skill.nc, residual_summary.md,
         FIGURES_DIR/diagnostics/residual_analysis.png

Usage:
  python scripts/09_residual_analysis.py
  python scripts/09_residual_analysis.py --no-maps        # skip loading the SST fields
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl, residual as rs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-file", type=Path,
                   default=paths.CESM2LE_DIR / "slowdowns" / "cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2030)
    p.add_argument("--no-maps", action="store_true"); p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "residual"; out_dir.mkdir(parents=True, exist_ok=True)
    years = np.arange(a.start_year, a.end_year + 1)
    with xr.open_dataset(a.labels_file) as ds:
        window = int(ds.attrs.get("window", 10))
        trend = ds["trend_anom"].sel(nyr=slice(a.start_year, a.end_year)).values.astype(float)
    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    print(f"09  —  residual analysis  onsets {a.start_year}–{a.end_year}, window {window} yr, "
          f"{np.isfinite(trend).sum()} member-years")

    # indices: onset year and mean over the trend decade
    idx_long = bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, a.start_year, a.end_year + window - 1)
    yrs_long = np.arange(a.start_year, a.end_year + window)
    idx_onset = {k: v[:, :years.size] for k, v in idx_long.items()}
    idx_conc = {k: rs.window_mean(v, yrs_long, years, window) for k, v in idx_long.items()}

    ds = rs.residual_skill(trend, sie_anom, idx_onset, idx_conc, years)
    resid, r2_pooled = rs.pooled_residual(trend, sie_anom)
    ds.to_netcdf(out_dir / "residual_skill.nc")
    md = rs.summary_markdown(ds, r2_pooled)
    (out_dir / "residual_summary.md").write_text(md); print("\n" + md)

    if a.no_maps:
        return
    from src.cnn.splits import load_jja_sst_demeaned
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        land = np.array(d["landmask"][:]) == 1
    print("\nloading JJA SST (onset year) ...")
    sst_onset, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=a.start_year,
                                         end_year=a.end_year, demean="group")
    r_onset = np.where(land, np.nan, rs.correlation_map(resid, sst_onset)); del sst_onset
    print(f"loading JJA SST ({window}-yr running mean) ...")
    sst_conc, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=a.start_year,
                                        end_year=a.end_year, demean="group", sst_window=window)
    r_conc = np.where(land, np.nan, rs.correlation_map(resid, sst_conc)); del sst_conc
    xr.Dataset({"r_onset": (("lat", "lon"), r_onset), "r_conc": (("lat", "lon"), r_conc)},
               coords={"lat": lat, "lon": lon}).to_netcdf(out_dir / "residual_corr_maps.nc")
    for name, r in (("onset", r_onset), ("concurrent", r_conc)):
        arc = np.nanmean(np.abs(r[lat >= 65])); trop = np.nanmean(np.abs(r[(lat > -20) & (lat < 20)]))
        print(f"  |r| {name}: Arctic mean {arc:.3f}, tropics mean {trop:.3f}, max {np.nanmax(np.abs(r)):.3f}")
    if not a.no_fig:
        from src.plotting import residual as plot, style as st
        st.paper_rc()
        plot.plot_residual(ds, trend, sie_anom, r_onset, r_conc, lat, lon,
                           paths.FIGURES_DIR / "diagnostics" / "residual_analysis.png", window)


if __name__ == "__main__":
    main()
