"""
02_cesm2le_slowdowns_relative.py — epoch-free slowdown labels (revision plan §4.5).

Defines a slowdown as a member's decadal-trend anomaly relative to its
forcing-group mean trend exceeding +n_sigma (pooled σ over a reference
period). Removes the year-dependence of the base rate caused by scaling an
observed threshold by a near-zero ensemble-mean trend. Original labels from
02_cesm2le_slowdowns.py are left untouched.

Outputs (one per window × sigma × sigma-mode):
  CESM2LE_SLOWDOWNS_DIR/cesm2le_{var}_slowdown_relative_{MON}_w{window}_s{sigma}_{demean}[_yearly]_1990-2100.nc
  FIGURES_DIR/<same stem>.png                 3-panel diagnostic per label file
  FIGURES_DIR/..._window_sweep.png            frequency by onset year across windows
  FIGURES_DIR/..._sigma_modes.png             pooled vs yearly σ (with --sigma-mode both; step 1.2)

Usage:
  python scripts/02_cesm2le_slowdowns_relative.py                  # 10-yr, 1σ, group demean
  python scripts/02_cesm2le_slowdowns_relative.py --window 3 5 7 10 15 --n-sigma 1.0
  python scripts/02_cesm2le_slowdowns_relative.py --demean all     # full-ensemble reference
  python scripts/02_cesm2le_slowdowns_relative.py --sigma-mode both   # step 1.2 decision figure
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import xarray as xr

from configs import paths
from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.data.cesm2le import slowdowns_relative as rel
from src.plotting import slowdowns as plot
from src.plotting import style as st


def relative_label_file(variable: str, month: str, window: int, n_sigma: float,
                        demean: str, start_year: int = 1990, end_year: int = 2100,
                        sigma_mode: str = "pooled", trend_offset: int = 0) -> Path:
    """Path of a relative-label file (mirrors paths.cesm2le_slowdown_file naming)."""
    mode = "" if sigma_mode == "pooled" else f"_{sigma_mode}"
    off = f"_off{trend_offset}" if trend_offset else ""
    return paths.CESM2LE_SLOWDOWNS_DIR / (
        f"cesm2le_{variable}_slowdown_relative_{month}_w{window}_s{n_sigma:g}_{demean}{mode}{off}"
        f"_{start_year}-{end_year}.nc")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable", default="sie", choices=["sie", "sia"])
    p.add_argument("--month", default="SEP")
    p.add_argument("--window", type=int, nargs="+", default=[10],
                   help="trend window length(s) in years")
    p.add_argument("--n-sigma", type=float, nargs="+", default=[1.0])
    p.add_argument("--demean", default="group", choices=["group", "all"],
                   help="reference trend: per forcing group (default) or full ensemble")
    p.add_argument("--sigma-mode", default="pooled", choices=["pooled", "yearly", "both"],
                   help="one σ over --pool-years (default), a smoothed σ per onset year, "
                        "or both (also draws the comparison figure)")
    p.add_argument("--cap-year", type=int, default=2030,
                   help="onset cap marked in the comparison figure (default 2030)")
    p.add_argument("--pool-years", type=int, nargs=2, default=[1990, 2040])
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2100)
    p.add_argument("--trend-offset", type=int, default=0,
                   help="label onset t with the trend of the window t+k…t+k+w−1 (default 0 = LB22 convention; "
                        "1 keeps the onset year out of the fitted window, step 8.9)")
    p.add_argument("--no-fig", action="store_true", help="skip diagnostic figures")
    p.add_argument("--member", type=int, default=7, help="member highlighted in the figure")
    return p.parse_args()


def main():
    st.paper_rc()
    a = parse_args()
    print("02  —  Relative (epoch-free) slowdown labels")
    sie, years = load_sie_monthly_files(str(paths.CESM2LE_AICE_DIR / "metrics"), a.month,
                                        variable=a.variable, start_year=a.start_year,
                                        end_year=a.end_year)
    print(f"  {a.variable.upper()} {a.month}: {sie.shape[0]} members, {years[0]}–{years[-1]}\n")
    paths.CESM2LE_SLOWDOWNS_DIR.mkdir(parents=True, exist_ok=True)
    orig_file = paths.cesm2le_slowdown_file(a.variable, a.month)
    original = xr.open_dataset(orig_file) if orig_file.exists() else None
    if original is None:
        print(f"  [note] original labels not found ({orig_file.name}); panel (c) shows relative only")
    sweep = {}
    modes = ["pooled", "yearly"] if a.sigma_mode == "both" else [a.sigma_mode]

    for w in a.window:
        for s in a.n_sigma:
            built = {}
            for mode in modes:
                ds = rel.build_relative_dataset(sie, years, window=w, start_year=a.start_year,
                                                trend_offset=a.trend_offset,
                                                n_sigma=s, demean=a.demean,
                                                pool_years=tuple(a.pool_years),
                                                sigma_mode=mode)
                out = relative_label_file(a.variable, a.month, w, s, a.demean,
                                          a.start_year, a.end_year, sigma_mode=mode,
                                          trend_offset=a.trend_offset)
                ds.to_netcdf(out)
                built[mode] = ds
                lab, yrs = ds["slowdown"].values, ds["nyr"].values
                sel = (yrs >= a.pool_years[0]) & (yrs <= a.pool_years[1])
                print(f"window {w:>2d} yr, {s:g}σ, σ-mode {mode}  →  {out.name}")
                sig = ds["sigma"].values
                print(f"  σ = {sig[sel].min():.4f}–{sig[sel].max():.4f} M km² yr⁻¹ over the pooling years"
                      if mode == "yearly" else f"  σ_pool = {float(sig[0]):.4f} M km² yr⁻¹")
                print("  slowdown frequency by decade (onset years "
                      f"{a.pool_years[0]}–{a.pool_years[1]}):")
                print(rel.frequency_table(lab[:, sel], yrs[sel]))
                if not a.no_fig:
                    plot.plot_relative_labels(ds, sie, years, paths.FIGURES_DIR / f"{out.stem}.png",
                                             original=original, member=a.member,
                                             pool_years=tuple(a.pool_years))
                print()
            if len(built) == 2 and not a.no_fig:
                stem = relative_label_file(a.variable, a.month, w, s, a.demean,
                                           a.start_year, a.end_year).stem
                plot.plot_sigma_mode_comparison(built["pooled"], built["yearly"], sie, years,
                                                paths.FIGURES_DIR / f"{stem}_sigma_modes.png",
                                                pool_years=tuple(a.pool_years), cap_year=a.cap_year)
                n_diff = int((built["pooled"]["slowdown"].values != built["yearly"]["slowdown"].values)[:, sel].sum())
                print(f"  pooled vs yearly: {n_diff} of {100 * int(sel.sum())} training-window labels differ\n")
            if s == a.n_sigma[0]:
                sweep[w] = built[modes[0]]

    if not a.no_fig and len(sweep) > 1:
        plot.plot_window_sweep(sweep, paths.FIGURES_DIR /
                              f"cesm2le_{a.variable}_slowdown_relative_{a.month}_window_sweep.png",
                              pool_years=tuple(a.pool_years))


if __name__ == "__main__":
    main()
