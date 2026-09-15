#!/usr/bin/env python3
"""
09_ohc_ledger.py — does upper-ocean heat at onset carry decadal memory beyond the target's own state?

cmip6 group (50 members), offset window t+1 … t+10. Stage 1 regresses the decadal trend anomaly on the state
at onset; stage 2 asks how much test R² OHC at onset adds — regional means, the map as K PCs, and the IPO for
reference — plus the pointwise correlation of the stage-1 residual with the OHC anomaly at onset.
  --target sie  September SIE; state = SIE anomaly (+ September volume); map north of 40°N; default JJA OHC
  --target gmt  GMT (the LB22 problem); state = GMT anomaly; global map; default annual OHC

Outputs: results/ohc/<target>_ledger_<depth>_<months>/ledger.csv, summary.md,
         FIGURES_DIR/diagnostics/ohc_<target>_ledger_<depth>_<months>.png

Usage:
  python scripts/09_ohc_ledger.py --depth 100                      # SIE, JJA
  python scripts/09_ohc_ledger.py --target gmt --depth 100 --cv --n-pcs 10 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl
from src.analysis import ohc as A
from src.data.cesm2le.slowdowns_gmt import load_gmt_yearly


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["sie", "gmt"], default="sie")
    p.add_argument("--depth", type=int, default=100)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029)
    p.add_argument("--window", type=int, default=10); p.add_argument("--offset", type=int, default=1)
    p.add_argument("--n-pcs", type=int, nargs="+", default=[10])
    p.add_argument("--months", type=int, nargs="+", default=None, help="months averaged for the OHC predictor (default JJA for sie, annual for gmt)")
    p.add_argument("--cv", action="store_true"); p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if a.months is None:
        a.months = [6, 7, 8] if a.target == "sie" else list(range(1, 13))
    tag = "annual" if sorted(a.months) == list(range(1, 13)) else "".join("JFMAMJJASOND"[m - 1] for m in sorted(a.months))
    out_dir = paths.RESULTS_DIR / "ohc" / f"{a.target}_ledger_{a.depth}_{tag}"; out_dir.mkdir(parents=True, exist_ok=True)
    onsets = np.arange(a.start_year, a.end_year + 1)
    ohc, years, lat, lon = A.load_ohc(paths.CESM2LE_DIR / "ohc" / f"ohc{a.depth}_cesmle_first50members_mon_1990-2100.nc", a.months)
    anom = A.demean(ohc); nens = ohc.shape[0]
    idx = np.searchsorted(years, onsets)

    if a.target == "sie":
        series = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", np.arange(1990, 2101), demean="group")[0][:nens]
        state = {"sie": A.demean(series)[:, idx]}
        siv = bl.load_siv_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")
        if siv is not None:
            state["siv"] = siv[:nens][:, idx]
        state_label, lat_min = "ice state", 40
    else:
        series, gyears = load_gmt_yearly(str(paths.CESM2LE_TREF_DIR / "gmt"), member_groups=["first50"], start_year=1990, end_year=2100)
        assert np.array_equal(years, gyears), "OHC and GMT years differ"
        state = {"gmt": A.demean(series)[:, idx]}
        state_label, lat_min = "GMT state", -90
    indices = bl.load_climate_indices_jja(paths.CESM2LE_CLIMATE_INDICES_DIR, a.start_year, a.end_year)
    ipo = indices["ipo"][:nens] if "ipo" in indices else None

    y = A.trend_anomaly(series, years, onsets, a.window, a.offset)
    ok_year = np.isfinite(y).all(0)
    regions = A.region_scalars(anom[:, idx], lat, lon)
    north = (lat >= lat_min)[:, None] & np.ones((1, lon.size), bool)
    ocean_n = (north & np.isfinite(ohc[0, 0])).ravel()
    map_name = f"OHC map >{lat_min}N" if lat_min > -90 else "OHC map (global)"
    maps = anom[:, idx].reshape(nens, onsets.size, -1)

    rows, base = [], []
    for tr, te in A.splits(nens, single=not a.cv):
        ytr, yte = A.flatten(y, tr, ok_year).ravel(), A.flatten(y, te, ok_year).ravel()
        S_tr = np.column_stack([A.flatten(v, tr, ok_year) for v in state.values()])
        S_te = np.column_stack([A.flatten(v, te, ok_year) for v in state.values()])
        r2_base, _ = A.fit_score(S_tr, ytr, S_te, yte, "regression", alpha=1e-3); base.append(r2_base)
        cands = {k: (A.flatten(v, tr, ok_year), A.flatten(v, te, ok_year)) for k, v in regions.items()}
        if ipo is not None:
            cands["IPO (reference)"] = (A.flatten(ipo, tr, ok_year), A.flatten(ipo, te, ok_year))
        Xtr_map, Xte_map = A.flatten(maps, tr, ok_year)[:, ocean_n], A.flatten(maps, te, ok_year)[:, ocean_n]
        for k in a.n_pcs:
            Ptr, Pte, _ = A.fit_pcs(Xtr_map, Xte_map, k); cands[f"{map_name}, {k} PCs"] = (Ptr, Pte)
        cands["all regions"] = (np.column_stack([cands[k][0] for k in regions]), np.column_stack([cands[k][1] for k in regions]))
        for name, (Xtr, Xte) in cands.items():
            r2, _ = A.fit_score(np.column_stack([S_tr, Xtr]), ytr, np.column_stack([S_te, Xte]), yte, "regression", alpha=1.0)
            rows.append(dict(block=int(te[0] // 10), predictor=name, r2_full=r2, delta=r2 - r2_base))
        print(f"block {int(te[0] // 10)}: {state_label} R² {r2_base:.3f}; " + "  ".join(f"{r['predictor']} +{r['delta']:+.3f}" for r in rows if r['block'] == int(te[0] // 10)))

    df = pd.DataFrame(rows); df.to_csv(out_dir / "ledger.csv", index=False)
    med = df.groupby("predictor")["delta"].median().sort_values(ascending=False)
    lines = [f"# OHC{a.depth} ({tag}) at onset and the decadal {a.target.upper()} trend (cmip6, 50 members, offset {a.offset}, {a.window}-yr windows)\n",
             f"Stage 1 ({state_label}: {', '.join(state)}) test R² = {np.nanmedian(base):.3f}. Added test R² from OHC predictors at onset (median over blocks):\n",
             "| predictor | ΔR² |", "|---|---|"] + [f"| {k} | {v:+.3f} |" for k, v in med.items()]
    md = "\n".join(lines) + "\n"; (out_dir / "summary.md").write_text(md); print("\n" + md)

    # residual–OHC correlation map (pooled, all members, single stage-1 OLS on the state)
    S_all = np.column_stack([v[:, ok_year].reshape(-1) for v in state.values()]); y_all = y[:, ok_year].reshape(-1)
    beta = np.linalg.lstsq(np.column_stack([S_all, np.ones(S_all.shape[0])]), y_all, rcond=None)[0]
    resid = y_all - np.column_stack([S_all, np.ones(S_all.shape[0])]) @ beta
    M = maps[:, ok_year].reshape(-1, maps.shape[-1])
    Mc = M - np.nanmean(M, 0); rc = resid - resid.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (rc @ np.nan_to_num(Mc)) / (np.sqrt((rc ** 2).sum()) * np.sqrt(np.nansum(Mc ** 2, 0)))
    corr = corr.reshape(lat.size, lon.size); corr[~np.isfinite(ohc[0, 0])] = np.nan
    xr.DataArray(corr, coords={"lat": lat, "lon": lon}, dims=("lat", "lon")).to_netcdf(out_dir / "residual_ohc_corr.nc")

    if not a.no_fig:
        from src.plotting import ohc as plot, style as st
        st.paper_rc()
        delta = {k: df[df.predictor == k]["delta"].values for k in med.index}
        plot.plot_ledger(delta, corr, lat, lon, paths.FIGURES_DIR / "diagnostics" / f"ohc_{a.target}_ledger_{a.depth}_{tag}.png",
                         float(np.nanmedian(base)), state_label)


if __name__ == "__main__":
    main()
