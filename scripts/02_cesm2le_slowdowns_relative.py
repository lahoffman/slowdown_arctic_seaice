"""
02_cesm2le_slowdowns_relative.py — epoch-free slowdown labels (revision plan §4.5).

Defines a slowdown as a member's decadal-trend anomaly relative to its
forcing-group mean trend exceeding +n_sigma (pooled σ over a reference
period). Removes the year-dependence of the base rate caused by scaling an
observed threshold by a near-zero ensemble-mean trend. Original labels from
02_cesm2le_slowdowns.py are left untouched.

Outputs (one per window × sigma):
  CESM2LE_SLOWDOWNS_DIR/cesm2le_{var}_slowdown_relative_{MON}_w{window}_s{sigma}_{demean}_1990-2100.nc
  FIGURES_DIR/<same stem>.png          3-panel diagnostic per label file
  FIGURES_DIR/..._window_sweep.png     frequency by onset year across windows

Usage:
  python scripts/02_cesm2le_slowdowns_relative.py                  # 10-yr, 1σ, group demean
  python scripts/02_cesm2le_slowdowns_relative.py --window 3 5 7 10 15 --n-sigma 1.0
  python scripts/02_cesm2le_slowdowns_relative.py --demean all     # full-ensemble reference
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
                        demean: str, start_year: int = 1990, end_year: int = 2100) -> Path:
    """Path of a relative-label file (mirrors paths.cesm2le_slowdown_file naming)."""
    return paths.CESM2LE_SLOWDOWNS_DIR / (
        f"cesm2le_{variable}_slowdown_relative_{month}_w{window}_s{n_sigma:g}_{demean}"
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
    p.add_argument("--sigma-mode", default="pooled", choices=["pooled", "yearly"])
    p.add_argument("--pool-years", type=int, nargs=2, default=[1990, 2040])
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2100)
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

    for w in a.window:
        for s in a.n_sigma:
            ds = rel.build_relative_dataset(sie, years, window=w, start_year=a.start_year,
                                            n_sigma=s, demean=a.demean,
                                            pool_years=tuple(a.pool_years),
                                            sigma_mode=a.sigma_mode)
            out = relative_label_file(a.variable, a.month, w, s, a.demean,
                                      a.start_year, a.end_year)
            ds.to_netcdf(out)
            lab, yrs = ds["slowdown"].values, ds["nyr"].values
            sel = (yrs >= a.pool_years[0]) & (yrs <= a.pool_years[1])
            print(f"window {w:>2d} yr, {s:g}σ  →  {out.name}")
            print(f"  σ_pool = {float(ds['sigma'][0]):.4f} M km² yr⁻¹")
            print("  slowdown frequency by decade (onset years "
                  f"{a.pool_years[0]}–{a.pool_years[1]}):")
            print(rel.frequency_table(lab[:, sel], yrs[sel]))
            if not a.no_fig:
                plot.plot_relative_labels(ds, sie, years, paths.FIGURES_DIR / f"{out.stem}.png",
                                         original=original, member=a.member,
                                         pool_years=tuple(a.pool_years))
            if s == a.n_sigma[0]:
                sweep[w] = ds
            print()

    if not a.no_fig and len(sweep) > 1:
        plot.plot_window_sweep(sweep, paths.FIGURES_DIR /
                              f"cesm2le_{a.variable}_slowdown_relative_{a.month}_window_sweep.png",
                              pool_years=tuple(a.pool_years))


if __name__ == "__main__":
    main()
