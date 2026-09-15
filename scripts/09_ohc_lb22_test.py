#!/usr/bin/env python3
"""
09_ohc_lb22_test.py — Labe & Barnes (2022) with their predictor and our controls (AIES §9).

Annual OHC100 anomaly maps (cmip6 group, 50 members) → decadal GMT-trend slowdown, for a window
starting at the onset year (their construction) and one year later (offset). Predictors, all scored
on held-out members: GMT state at onset; global-mean OHC100 at onset; OHC100 map as K PCs (ridge /
logistic); OHC100 PCs + GMT state; optionally a small LB22-style MLP on the PCs. Also reports
r(OHC-map fit, GMT state) — the proxy strength that sets the coupling's share (0.25 r²).
The same table is produced for September SIE as the target, so the two problems sit side by side.

Outputs: results/ohc/lb22_test/lb22_test.nc, summary.md, FIGURES_DIR/diagnostics/ohc_lb22_test.png

Usage:
  python scripts/09_ohc_lb22_test.py                        # single split (last 10 members), 20 PCs
  python scripts/09_ohc_lb22_test.py --cv --n-pcs 20 50 --ann
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
from src.analysis import baselines as bl
from src.analysis import ohc as A
from src.data.cesm2le.slowdowns_gmt import load_gmt_yearly


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ohc-file", type=Path, default=paths.CESM2LE_DIR / "ohc" / "ohc100_cesmle_first50members_mon_1990-2100.nc")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2029)
    p.add_argument("--window", type=int, default=10); p.add_argument("--offsets", type=int, nargs="+", default=[0, 1])
    p.add_argument("--n-pcs", type=int, nargs="+", default=[20])
    p.add_argument("--months", type=int, nargs="+", default=list(range(1, 13)), help="months averaged for the OHC predictor (default annual, as LB22)")
    p.add_argument("--cv", action="store_true", help="all five member blocks instead of the last one")
    p.add_argument("--ann", action="store_true", help="add an LB22-style MLP (2×30) on the PCs, one seed")
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "ohc" / "lb22_test"; out_dir.mkdir(parents=True, exist_ok=True)
    onsets = np.arange(a.start_year, a.end_year + 1)

    ohc, oyears, lat, lon = A.load_ohc(a.ohc_file, a.months)
    ohc_anom = A.demean(ohc)
    ocean = np.isfinite(ohc[0, 0]).ravel()
    gmt, gyears = load_gmt_yearly(str(paths.CESM2LE_TREF_DIR / "gmt"), member_groups=["first50"], start_year=1990, end_year=2100)
    sie_all, _ = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", np.arange(1990, 2101), demean="group")
    sie = sie_all[:50]
    assert np.array_equal(oyears, gyears), "OHC and GMT years differ"
    nens = ohc.shape[0]

    # member ordering check: global-mean OHC vs GMT anomaly
    gm_ohc = A.area_mean(ohc_anom, lat, lon, A.REGIONS["global"])
    r_match, r_mis = A.member_order_check(gm_ohc, A.demean(gmt))
    print(f"member-order check: matched r = {r_match:.2f}, mismatched r = {r_mis:.2f}  (matched must be clearly larger)")

    idx = np.searchsorted(oyears, onsets)
    maps = ohc_anom[:, idx].reshape(nens, onsets.size, -1)                 # (nens, nonset, cells)
    rows = []
    for target_name, series, sign in (("GMT", gmt, -1), ("SIE", sie, +1)):
        state = A.demean(series)[:, idx]
        for off in a.offsets:
            y_cont = A.trend_anomaly(series, oyears, onsets, a.window, off)
            y_bin = A.binary_labels(y_cont, sign)
            ok_year = np.isfinite(y_cont).all(0)
            for tr, te in A.splits(nens, single=not a.cv):
                yc_tr, yc_te = A.flatten(y_cont, tr, ok_year).ravel(), A.flatten(y_cont, te, ok_year).ravel()
                yb_tr, yb_te = A.flatten(y_bin, tr, ok_year).ravel(), A.flatten(y_bin, te, ok_year).ravel()
                feats = {"state": (A.flatten(state, tr, ok_year), A.flatten(state, te, ok_year)),
                         "ohc_global": (A.flatten(gm_ohc[:, idx], tr, ok_year), A.flatten(gm_ohc[:, idx], te, ok_year))}
                Xtr_map, Xte_map = A.flatten(maps, tr, ok_year)[:, ocean], A.flatten(maps, te, ok_year)[:, ocean]
                for k in a.n_pcs:
                    Ptr, Pte, _ = A.fit_pcs(Xtr_map, Xte_map, k)
                    feats[f"ohc_map_pc{k}"] = (Ptr, Pte)
                    feats[f"ohc_map_pc{k}+state"] = (np.column_stack([Ptr, feats["state"][0]]), np.column_stack([Pte, feats["state"][1]]))
                for fname, (Xtr, Xte) in feats.items():
                    r2, p_cont = A.fit_score(Xtr, yc_tr, Xte, yc_te, "regression")
                    au, _ = A.fit_score(Xtr, yb_tr, Xte, yb_te, "classification")
                    # proxy strength: how much of the fitted map prediction is the current state
                    r_state = float(np.corrcoef(p_cont, feats["state"][1].ravel())[0, 1]) if fname.startswith("ohc_map") else np.nan
                    rows.append(dict(target=target_name, offset=off, test_block=int(te[0] // 10), predictor=fname,
                                     r2=r2, auroc=au, r_pred_state=r_state))
                    if a.ann and fname.startswith("ohc_map_pc") and "+state" not in fname:
                        rows.append(dict(target=target_name, offset=off, test_block=int(te[0] // 10), predictor=fname + "_ann",
                                         r2=A.ann_score(Xtr, yc_tr, Xte, yc_te, "regression"),
                                         auroc=A.ann_score(Xtr, yb_tr, Xte, yb_te, "classification"), r_pred_state=np.nan))
                print(f"{target_name} offset {off} block {int(te[0] // 10)}: " +
                      "  ".join(f"{r['predictor']} R²={r['r2']:.3f} AUROC={r['auroc']:.3f}" for r in rows
                               if r["target"] == target_name and r["offset"] == off and r["test_block"] == int(te[0] // 10)))

    import pandas as pd
    df = pd.DataFrame(rows)
    med = df.groupby(["target", "offset", "predictor"])[["r2", "auroc", "r_pred_state"]].median().reset_index()
    med.to_csv(out_dir / "lb22_test.csv", index=False)
    xr.Dataset.from_dataframe(df.set_index(["target", "offset", "predictor", "test_block"])).to_netcdf(out_dir / "lb22_test.nc")
    lines = ["# LB22 with their predictor and our controls — OHC100 (cmip6, 50 members), member-block test scores (median over blocks)\n",
             f"Onsets {a.start_year}–{a.end_year}, {a.window}-yr windows. r_pred_state = corr(map-based prediction, target state at onset).\n"]
    for tgt in ("GMT", "SIE"):
        lines += [f"\n## target: {tgt} slowdown\n", "| predictor | " + " | ".join(f"R² off{o} | AUROC off{o}" for o in a.offsets) + " | r(pred, state) |",
                  "|---|" + "---|---|" * len(a.offsets) + "---|"]
        for pred in med[med.target == tgt].predictor.unique():
            cells = []
            for o in a.offsets:
                r = med[(med.target == tgt) & (med.offset == o) & (med.predictor == pred)]
                cells += [f"{float(r.r2.iloc[0]):.3f}", f"{float(r.auroc.iloc[0]):.3f}"] if len(r) else ["—", "—"]
            rs = med[(med.target == tgt) & (med.offset == a.offsets[0]) & (med.predictor == pred)].r_pred_state
            r_txt = f"{float(rs.iloc[0]):.2f}" if len(rs) and np.isfinite(rs.iloc[0]) else "—"
            lines.append(f"| {pred} | " + " | ".join(cells) + f" | {r_txt} |")
    lines.append(f"\nmember-order check: matched r {r_match:.2f} vs mismatched {r_mis:.2f}.\n")
    md = "\n".join(lines) + "\n"; (out_dir / "summary.md").write_text(md); print("\n" + md)

    if not a.no_fig:
        from src.plotting import ohc as plot, style as st
        st.paper_rc()
        plot.plot_lb22_test(med, a.offsets, paths.FIGURES_DIR / "diagnostics" / "ohc_lb22_test.png")


if __name__ == "__main__":
    main()
