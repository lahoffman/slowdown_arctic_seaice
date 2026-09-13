"""
09_obs_forced_removal.py — how the observed Arctic SST anomaly depends on the
forced reference and the product (step 6.1).

Arctic (>65°N) JJA SST from ERSSTv5 and OISST v2.1 on the CESM2 grid, with
the forced part removed by: linear trend, quadratic trend, 100-member mean,
CMIP6-BB group mean, SMBB group mean (model references mean- and
trend-corrected to the observations, as in the CNN input pipeline). The sign
of the 2016–2025 anomaly — the years the paper's observational claim rests on
— is tabulated for every combination.

Outputs: results/obs_forced/arctic_index_methods.nc, summary.md,
         FIGURES_DIR/diagnostics/obs_forced_removal.png

Usage:
  python scripts/09_obs_forced_removal.py [--end-year 2025]
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
from src.data.observations import obs_input as oi
from src.data.observations.ersst.climate_indices import _correct_forced_mean_and_trend
from src.plotting.forced import arctic_mean

METHODS = ["linear", "quadratic", "ensmean", "group_cmip6", "group_smbb"]


def residual_series(obs: np.ndarray, years: np.ndarray, method: str, forced_arctic: dict) -> np.ndarray:
    t = years - years.mean()
    if method == "linear":
        return obs - np.polyval(np.polyfit(t, obs, 1), t)
    if method == "quadratic":
        return obs - np.polyval(np.polyfit(t, obs, 2), t)
    _, _, resid = _correct_forced_mean_and_trend(obs, forced_arctic[method], np.arange(years.size, dtype=float))
    return resid


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--claim-years", type=int, nargs=2, default=[2016, 2025])
    p.add_argument("--no-fig", action="store_true")
    a = p.parse_args()
    out_dir = paths.RESULTS_DIR / "obs_forced"; out_dir.mkdir(parents=True, exist_ok=True)
    years = np.arange(a.start_year, a.end_year + 1)
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat = np.array(g["lat"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        landmask = np.array(d["landmask"][:])
    mask = lambda f: np.where(landmask == 1, np.nan, f)

    forced_arctic = {m: arctic_mean(mask(oi.forced_sst_field(m, years, paths.CESM2LE_ENSMEAN_JJA,
                                                              paths.CESM2LE_GROUPMEAN_JJA)), lat)
                     for m in ("ensmean", "group_cmip6", "group_smbb")}
    series, prods = {}, {}
    for name, path in paths.OBS_PRODUCTS.items():
        if not path.exists():
            print(f"  [skip] {name}: {path} missing"); continue
        prod = oi.load_monthly_product(path)
        jja = mask(oi.jja_by_year(prod, years))
        ok = ~np.isnan(jja).all(axis=(1, 2))
        if ok.sum() < 10:
            print(f"  [skip] {name}: only {int(ok.sum())} years with JJA data in {years[0]}–{years[-1]}"); continue
        if not ok.all():
            print(f"  [note] {name}: no JJA for {years[~ok]} — truncating")
        yrs = years[ok]; obs = arctic_mean(jja[ok], lat)
        prods[name] = (yrs, obs)
        series[name] = {m: residual_series(obs, yrs, m, {k: v[ok] for k, v in forced_arctic.items()}) for m in METHODS}

    c0, c1 = a.claim_years
    lines = ["| product | method | mean anomaly " + f"{c0}–{c1} [°C] | years > 0 | 2010–2015 mean |", "|---|---|---|---|---|"]
    for name, res in series.items():
        yrs = prods[name][0]; sel = (yrs >= c0) & (yrs <= c1); sel2 = (yrs >= 2010) & (yrs <= 2015)
        for m in METHODS:
            r = res[m]
            lines.append(f"| {name} | {m} | {r[sel].mean():+.3f} | {int((r[sel] > 0).sum())}/{int(sel.sum())} | {r[sel2].mean():+.3f} |")
    md = ("# Observed Arctic (>65°N) JJA SST anomaly by forced reference and product\n\n"
          "Model references are mean- and trend-corrected to the observations before subtraction "
          "(same as the CNN input pipeline).\n\n" + "\n".join(lines) + "\n")
    (out_dir / "summary.md").write_text(md); print(md)
    ds = xr.Dataset({f"{n}_{m}": ("year", np.interp(years, prods[n][0], series[n][m], left=np.nan, right=np.nan))
                     for n in series for m in METHODS} | {f"{n}_raw": ("year", np.interp(years, *prods[n], left=np.nan, right=np.nan)) for n in series},
                    coords={"year": years})
    ds.to_netcdf(out_dir / "arctic_index_methods.nc")
    if not a.no_fig and series:
        from src.plotting import observations as plot_obs, style as st
        st.paper_rc()
        plot_obs.plot_forced_removal(prods, series, forced_arctic, years, METHODS, (c0, c1),
                                     paths.FIGURES_DIR / "diagnostics" / "obs_forced_removal.png")


if __name__ == "__main__":
    main()
