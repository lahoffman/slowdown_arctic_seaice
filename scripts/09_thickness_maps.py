#!/usr/bin/env python3
"""
09_thickness_maps.py — is the decadal memory in the thickness *pattern*? (step 8.12; linear look before any CNN)

Thickness (hi, CICE grid, north of 60°N) in one month of the onset year, demeaned per forcing
group, against the offset trend anomaly (t+1…t+10): correlation maps (raw and after removing
the SIE(t) fit), area-weighted EOFs, sector volumes, and out-of-sample R² / AUROC of scalar sets
(SIE; + pan-Arctic volume; + sector volumes; + leading PCs) on member-block splits. Minutes, CPU.

Outputs: results/thickness/<key>/thickness_summary.md, thickness_maps.nc,
         FIGURES_DIR/diagnostics/thickness_<MON>_<key>.png
Usage:
  python scripts/09_thickness_maps.py --labels-file $LBL1 --end-year 2029 --month SEP
  python scripts/09_thickness_maps.py --labels-file $LBL1 --end-year 2029 --month MAR --n-pcs 5 10 20
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
from src.analysis import baselines as bl, thickness as th
from src.analysis.residual import pooled_residual


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-file", type=Path, required=True)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029)
    p.add_argument("--month", default="SEP", help="thickness month of the onset year (SEP, MAR, ...)")
    p.add_argument("--n-pcs", type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--label-var", default="slowdown", choices=["slowdown", "riles"])
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    years = np.arange(a.start_year, a.end_year + 1)
    with xr.open_dataset(a.labels_file) as ds:
        key = f"w{int(ds.attrs.get('window', 10))}_off{int(ds.attrs.get('trend_offset', 0))}"
        trend = ds["trend_anom"].sel(nyr=slice(a.start_year, a.end_year)).values.astype(float)
        labels = ds[a.label_var].sel(nyr=slice(a.start_year, a.end_year)).values.astype(int)
    out_dir = paths.RESULTS_DIR / "thickness" / key; out_dir.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(paths.CESM2LE_CICE_GRID_FILE) as g:
        tlat, tlon, tarea = (np.array(g[v][:], float) for v in ("TLAT", "TLON", "tarea"))
    print(f"09  —  thickness maps  {a.month} of onset year, target {key}, onsets {a.start_year}–{a.end_year}")
    hi = th.load_hi_month(paths.CESM2LE_HI_MONTHLY, ["first50", "last50"], a.month, np.arange(1990, 2101), years)
    north = tlat >= 60
    hi_clim = np.where(north, hi.mean((0, 1)), np.nan)
    anom = th.group_demean(hi)
    ice_ever = (hi.max((0, 1)) > 0.05) & north                   # cells with ice at some point; others are constant
    del hi

    _, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
    resid, r2_pooled = pooled_residual(trend, sie_anom)
    r_trend = np.where(ice_ever, th.correlation_map(trend, anom), np.nan)
    r_resid = np.where(ice_ever, th.correlation_map(resid, anom), np.nan)
    vols = th.sector_volumes(anom, tlat, tlon, tarea)
    pan = th.sector_volumes(anom, tlat, tlon, tarea, {"pan": ((60, 90), (0, 360))})["pan"]
    w = np.cos(np.deg2rad(tlat)) * tarea / tarea.max()
    pcs, pats, evr = th.eofs(anom, w, ice_ever, n=max(a.n_pcs))
    print(f"  EOF explained variance (first 5): {np.round(100 * evr[:5], 1)} %")

    sets = {"SIE": [sie_anom], "SIE + pan vol": [sie_anom, pan],
            "SIE + sectors": [sie_anom] + list(vols.values())}
    for n in a.n_pcs:
        sets[f"SIE + {n} PCs"] = [sie_anom] + [pcs[..., i] for i in range(n)]
    sets["sectors only"] = list(vols.values()); sets[f"{max(a.n_pcs)} PCs only"] = [pcs[..., i] for i in range(max(a.n_pcs))]
    r2 = th.scalar_set_r2(trend, sets, years)
    au = th.scalar_set_auroc(labels, sets, years)

    med = lambda v: float(np.nanmedian(v))
    lines = [f"# Thickness ({a.month} of onset year) as predictor — target {key}, label `{a.label_var}`\n",
             f"Cells with ice: {int(ice_ever.sum())}; EOF variance first 5: {np.round(100 * evr[:5], 1)} %. "
             f"|corr(hi, trend)| max {np.nanmax(np.abs(r_trend)):.2f}; |corr(hi, residual)| max {np.nanmax(np.abs(r_resid)):.2f}.\n",
             "| scalar set | test R² (median) | range | test AUROC (median) |", "|---|---|---|---|"]
    for n in sets:
        lines.append(f"| {n} | {med(r2[n]):.3f} | {np.nanmin(r2[n]):.3f}–{np.nanmax(r2[n]):.3f} | {med(au[n]):.3f} |")
    lines.append("\nSector volume correlations with the residual (pooled):")
    for n, v in vols.items():
        ok = np.isfinite(resid); lines.append(f"- {n}: r = {np.corrcoef(v[ok], resid[ok])[0, 1]:+.3f}")
    md = "\n".join(lines) + "\n"
    (out_dir / f"thickness_summary_{a.month}.md").write_text(md); print("\n" + md)
    xr.Dataset({"hi_clim": (("nj", "ni"), hi_clim), "r_trend": (("nj", "ni"), r_trend), "r_resid": (("nj", "ni"), r_resid),
                "eof": (("mode", "nj", "ni"), pats), "evr": ("mode", evr),
                "TLAT": (("nj", "ni"), tlat), "TLON": (("nj", "ni"), tlon)}).to_netcdf(out_dir / f"thickness_maps_{a.month}.nc")
    if not a.no_fig:
        from src.plotting import thickness as plot, style as st
        st.paper_rc()
        plot.plot_thickness(hi_clim, r_trend, r_resid, pats[0], evr, r2, au, tlat, tlon,
                            paths.FIGURES_DIR / "diagnostics" / f"thickness_{a.month}_{key}.png", month=a.month)


if __name__ == "__main__":
    main()
