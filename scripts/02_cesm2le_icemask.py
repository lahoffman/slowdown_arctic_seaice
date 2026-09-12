"""
02_cesm2le_icemask.py — JJA sea-ice cover on the SST grid (revision step 1.6).

Nearest-neighbour regrids the JJA-mean CICE ice concentration of every member
and year onto the 192 x 288 atmosphere grid and saves the ice-cover mask
(aice > threshold) used by ``03_cesm2le_tvt_splits.py --openwater``.

Outputs:
  CESM2LE_ICEMASK_JJA                       icemask + aice_jja, (nens, year, nx, ny)
  FIGURES_DIR/diagnostics/icemask_summary.png

Usage:
  python scripts/02_cesm2le_icemask.py
  python scripts/02_cesm2le_icemask.py --threshold 0.15 --no-fig
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le import icemask as im

MEMBER_GROUPS = ["first50", "last50"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.15, help="ice-cover threshold (default 0.15)")
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2100)
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    years = np.arange(a.start_year, a.end_year + 1)
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as ds:
        lat, lon = np.array(ds["lat"][:]), np.array(ds["lon"][:])
    print("02  —  JJA ice-cover mask on the SST grid")
    print(f"  aice files : {paths.CESM2LE_AICE_DIR / 'mon'}")
    print(f"  CICE grid  : {paths.CESM2LE_CICE_GRID_FILE}")
    print(f"  threshold  : aice > {a.threshold:g}")
    ds = im.build_icemask(paths.CESM2LE_AICE_MONTHLY, MEMBER_GROUPS, paths.CESM2LE_CICE_GRID_FILE,
                          lat, lon, years, threshold=a.threshold)
    out = paths.CESM2LE_ICEMASK_JJA
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out, encoding={v: {"zlib": True, "complevel": 4} for v in ds.data_vars})
    arctic = lat >= 65
    frac = ds["icemask"].values[:, :, arctic, :].mean(axis=(0, 2, 3))
    print(f"  saved → {out}")
    print("  mean ice-covered fraction of cells north of 65°N by decade:")
    for y0 in range(a.start_year, a.end_year, 10):
        sel = (years >= y0) & (years < y0 + 10)
        print(f"    {y0}s  {frac[sel].mean():.2f}")
    if not a.no_fig:
        from src.plotting import icemask as plot, style as st
        st.paper_rc()
        landmask = None
        if paths.LANDMASK_FILE.exists():
            with nc.Dataset(paths.LANDMASK_FILE) as d:
                landmask = np.array(d["landmask"][:])
        plot.plot_icemask_summary(ds, paths.FIGURES_DIR / "diagnostics" / "icemask_summary.png",
                                  landmask=landmask)


if __name__ == "__main__":
    main()
