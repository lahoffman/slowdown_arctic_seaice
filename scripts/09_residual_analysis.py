#!/usr/bin/env python3
"""
09_residual_analysis.py — does SST explain the part of the trend the ice state does not? (step 8.1)

Continuous target: the 10-yr September SIE trend anomaly (relative to the forcing
group), no threshold. Stage 1 fits trend ~ SIE anomaly at onset; stage 2 fits the
residual on the SST indices, either at onset or averaged over the trend decade
(concurrent). Also maps the correlation of the residual with JJA SST at onset and
with the decade-mean JJA SST. Baselines only — minutes, no GPU.

Outputs: results/residual/w<window>_off<offset>/{residual_skill.nc, residual_summary.md, residual_corr_maps.nc},
         FIGURES_DIR/diagnostics/residual_analysis_w<window>_off<offset>.png

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
    p.add_argument("--siv-months", nargs="+", default=["SEP"],
                   help="months of sea-ice volume to test as extra ice-state scalars (e.g. SEP MAR); "
                        "MAR of the onset year precedes the September onset, so it is a legitimate predictor")
    p.add_argument("--sst-pcs", type=int, nargs="*", default=[10, 20],
                   help="PCs of the SST field (onset and decade-mean) as a stage-2 predictor set; empty to skip")
    p.add_argument("--pc-mask-north", type=float, default=65.0,
                   help="exclude SST poleward of this latitude from the PCs (Arctic SST is the ice state itself)")
    p.add_argument("--no-maps", action="store_true"); p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    years = np.arange(a.start_year, a.end_year + 1)
    with xr.open_dataset(a.labels_file) as ds:
        window = int(ds.attrs.get("window", 10)); offset = int(ds.attrs.get("trend_offset", 0))
        trend = ds["trend_anom"].sel(nyr=slice(a.start_year, a.end_year)).values.astype(float)
    key = f"w{window}_off{offset}"                       # one output folder per label definition
    out_dir = paths.RESULTS_DIR / "residual" / key; out_dir.mkdir(parents=True, exist_ok=True)
    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    print(f"09  —  residual analysis  onsets {a.start_year}–{a.end_year}, window {window} yr, offset {offset}, "
          f"{np.isfinite(trend).sum()} member-years")

    # indices: onset year, and mean over the trend decade t+offset … t+offset+window−1
    idx_long = bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, a.start_year, a.end_year + offset + window - 1)
    yrs_long = np.arange(a.start_year, a.end_year + offset + window)
    idx_onset = {k: v[:, :years.size] for k, v in idx_long.items()}
    idx_conc = {k: rs.window_mean(v, yrs_long, years + offset, window) for k, v in idx_long.items()}

    ds = rs.residual_skill(trend, sie_anom, idx_onset, idx_conc, years)
    resid, r2_pooled = rs.pooled_residual(trend, sie_anom)
    ds.to_netcdf(out_dir / "residual_skill.nc")
    md = rs.summary_markdown(ds, r2_pooled, window=window)

    # step 8.5: sea-ice volume (one or more months) as extra ice-state scalars, if 01 --variable hi has been run
    for mon in a.siv_months:
        siv_anom = bl.load_siv_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, month=mon, demean="group")
        if siv_anom is None:
            print(f"  [skip] no sivoln_*_{mon}.nc"); continue
        ds_v = rs.residual_skill(trend, siv_anom, idx_onset, idx_conc, years)
        ds_sv = rs.residual_skill(trend, sie_anom, idx_onset, idx_conc, years, extra_state={f"siv_{mon}": siv_anom})
        _, r2_sv = rs.pooled_residual(trend, sie_anom, siv_anom)
        ds_sv.to_netcdf(out_dir / f"residual_skill_sie_siv{mon}.nc")
        md += (f"\n## With {mon} sea-ice volume (step 8.5)\n\nStage-1 test R² (median): SIE {float(ds['r2_sie'].median()):.3f}, "
               f"{mon} volume alone {float(ds_v['r2_sie'].median()):.3f}, SIE + {mon} volume **{float(ds_sv['r2_sie'].median()):.3f}**.\n\n"
               + rs.summary_markdown(ds_sv, r2_sv, window=window).split("\n", 2)[2])
    # pooled residual for the tail-asymmetry script (09_tail_asymmetry.py)
    siv_sep = bl.load_siv_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, month="SEP", demean="group")
    state = [sie_anom] + ([siv_sep] if siv_sep is not None else [])
    resid_pooled, r2_state = rs.pooled_residual(trend, *state)
    fit_pooled = trend - resid_pooled
    xr.Dataset({"trend_anom": (("nens", "year"), trend), "fit_state": (("nens", "year"), fit_pooled),
                "residual": (("nens", "year"), resid_pooled), "sie_anom": (("nens", "year"), sie_anom),
                **({"siv_anom": (("nens", "year"), siv_sep)} if siv_sep is not None else {})},
               coords={"year": years}, attrs={"state": "sie_anom" + (" + siv_SEP" if siv_sep is not None else ""),
                                               "r2_pooled": r2_state, "window": window, "offset": offset}
               ).to_netcdf(out_dir / "residual_pooled.nc")

    if a.no_maps:
        (out_dir / "residual_summary.md").write_text(md); print("\n" + md); return
    from src.cnn.splits import load_jja_sst_demeaned
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        land = np.array(d["landmask"][:]) == 1
    print("\nloading JJA SST (onset year) ...")
    sst_onset, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=a.start_year,
                                         end_year=a.end_year, demean="group")
    r_onset = np.where(land, np.nan, rs.correlation_map(resid, sst_onset))
    sst_onset_keep = sst_onset if a.sst_pcs else None; del sst_onset
    print(f"loading JJA SST ({window}-yr running mean) ...")
    sst_conc, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=a.start_year + offset,
                                        end_year=a.end_year + offset, demean="group", sst_window=window)   # decade t+offset …
    r_conc = np.where(land, np.nan, rs.correlation_map(resid, sst_conc))
    if a.sst_pcs:
        cell_mask = (~land & (lat[:, None] < a.pc_mask_north) & np.ones((1, lon.size), bool)).ravel()
        ds_pc = rs.pc_skill(trend, state, sst_onset_keep, sst_conc, years, cell_mask, a.sst_pcs)
        ds_pc.to_netcdf(out_dir / "residual_skill_sst_pcs.nc")
        md += (f"\n## SST field as PCs (poleward of {a.pc_mask_north:g}°N excluded), ΔR² over the ice state ({', '.join(['sie'] + (['siv'] if siv_sep is not None else []))})\n\n"
               "| PCs | onset | concurrent (decade mean) |\n|---|---|---|\n" +
               "\n".join(f"| {k} | {float(ds_pc.delta_r2.sel(n_pcs=k, timing='onset').median()):+.3f} | "
                         f"{float(ds_pc.delta_r2.sel(n_pcs=k, timing='conc').median()):+.3f} |" for k in a.sst_pcs) + "\n")
        del sst_onset_keep
    del sst_conc
    (out_dir / "residual_summary.md").write_text(md); print("\n" + md)
    xr.Dataset({"r_onset": (("lat", "lon"), r_onset), "r_conc": (("lat", "lon"), r_conc)},
               coords={"lat": lat, "lon": lon}).to_netcdf(out_dir / "residual_corr_maps.nc")
    for name, r in (("onset", r_onset), ("concurrent", r_conc)):
        arc = np.nanmean(np.abs(r[lat >= 65])); trop = np.nanmean(np.abs(r[(lat > -20) & (lat < 20)]))
        print(f"  |r| {name}: Arctic mean {arc:.3f}, tropics mean {trop:.3f}, max {np.nanmax(np.abs(r)):.3f}")
    if not a.no_fig:
        from src.plotting import residual as plot, style as st
        st.paper_rc()
        plot.plot_residual(ds, trend, sie_anom, r_onset, r_conc, lat, lon,
                           paths.FIGURES_DIR / "diagnostics" / f"residual_analysis_{key}.png", window)


if __name__ == "__main__":
    main()
