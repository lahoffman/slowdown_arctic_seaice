"""
make_figure.py — build one manuscript figure from cached outputs.

Loading lives here (via configs.paths); drawing lives in src/plotting/paper.py;
figures are saved to FIGURES_DIR/paper/. Add a figure by writing
``paper.fig_<name>`` and a matching ``load_<name>`` below, then registering it
in FIGURES.

Usage:
  python scripts/make_figure.py S1                         # original labels
  python scripts/make_figure.py S1 --labels relative       # relative labels (default w10, 1σ, group)
  python scripts/make_figure.py S1 --labels relative --window 5 --member 12 --fmt pdf
  python scripts/make_figure.py --list
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.plotting import paper, style as st

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


# =============================================================================
# Loaders
# =============================================================================

def load_nsidc(variable: str, month: str) -> dict:
    """Observed series, trend windows, slowdown flags, mean and μ+σ threshold."""
    m = MONTHS.index(month)
    thr_file = paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if variable == "sie" else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS
    ev_file = (paths.nsidc_sie_slowdown_events if variable == "sie" else paths.nsidc_sia_slowdown_events)(m + 1)
    with xr.open_dataset(thr_file) as ds:
        thr = float(ds["threshold_slowdown"].values[m]); frac = float(ds["fraction_slowdown"].values[m])
    with xr.open_dataset(ev_file) as ds:
        out = {"trend_years": ds["year"].values.astype(int), "trends": ds["linear_trend"].values,
               "slowdown": ds["slowdown"].values.astype(int), "ice": ds["seaice"].values,
               "ice_years": ds.coords["yearice"].values.astype(int)}
    out.update(threshold=thr, mean_trend=thr / frac)
    return out


def label_file(kind: str, variable: str, month: str, window: int, n_sigma: float, demean: str) -> Path:
    if kind == "original":
        return paths.cesm2le_slowdown_file(variable, month)
    return paths.CESM2LE_SLOWDOWNS_DIR / (
        f"cesm2le_{variable}_slowdown_relative_{month}_w{window}_s{n_sigma:g}_{demean}_1990-2100.nc")


def load_s1(a):
    sie, years = load_sie_monthly_files(str(paths.CESM2LE_AICE_DIR / "metrics"), a.month,
                                        variable=a.variable, start_year=1990, end_year=2100)
    labels = xr.open_dataset(label_file(a.labels, a.variable, a.month, a.window, a.n_sigma, a.demean))
    return dict(nsidc=load_nsidc(a.variable, a.month), sie=sie, years=years, labels=labels,
                member=a.member, window=a.window if a.labels == "relative" else 10,
                varname=a.variable.upper(), month=a.month)


# =============================================================================
# Registry
# =============================================================================

FIGURES = {
    "S1": (load_s1, paper.fig_s1, "slowdown definition: NSIDC + CESM2-LE, 6 panels"),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figure", nargs="?", help="figure id, e.g. S1")
    p.add_argument("--list", action="store_true", help="list available figures")
    p.add_argument("--labels", default="original", choices=["original", "relative"])
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--n-sigma", type=float, default=1.0)
    p.add_argument("--demean", default="group", choices=["group", "all"])
    p.add_argument("--member", type=int, default=6)
    p.add_argument("--variable", default="sie", choices=["sie", "sia"])
    p.add_argument("--month", default="SEP")
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out", type=Path, default=None, help="explicit output path")
    return p.parse_args()


def main():
    a = parse_args()
    if a.list or a.figure is None:
        for k, (_, _, desc) in FIGURES.items():
            print(f"  {k:4s} {desc}")
        return
    loader, drawer, _ = FIGURES[a.figure]
    fig = drawer(**loader(a))
    tag = "" if a.labels == "original" else f"_rel_w{a.window}_s{a.n_sigma:g}_{a.demean}"
    out = a.out or paths.FIGURES_DIR / "paper" / f"fig_{a.figure}{tag}.{a.fmt}"
    st.save(fig, out, dpi=a.dpi)


if __name__ == "__main__":
    main()
