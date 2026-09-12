"""
02_obs_compare_products.py — ERSSTv5 vs OISST v2.1 on the CESM2-LE grid (step 6.2).

Both products are compared after regridding, over the years they share:
  (a) Arctic (>65°N) JJA-mean SST, each product, plus the difference
  (b) map of the JJA climatological difference OISST − ERSST
  (c) map of the difference in JJA linear trends OISST − ERSST
  (d) coverage: fraction of Arctic ocean cells with a valid SST in JJA, and the
      OISST ice fraction, to show what each product does under sea ice

Outputs: FIGURES_DIR/diagnostics/obs_products_compare.png and a printed table.

Usage:
  python scripts/02_obs_compare_products.py [--start-year 1990 --end-year 2025]
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
from src.analysis import obs_products as op
from src.plotting import observations as plot_obs
from src.plotting import style as st


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--clim", type=int, nargs=2, default=[1991, 2020])
    return p.parse_args()


def main():
    st.paper_rc()
    a = parse_args()
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    landmask = None
    if paths.LANDMASK_FILE.exists():
        with nc.Dataset(paths.LANDMASK_FILE) as d:
            landmask = np.array(d["landmask"][:])
    ersst = op.load_product(paths.ERSST_REGRIDDED, "ERSSTv5")
    oisst = op.load_product(paths.OISST_REGRIDDED, "OISST v2.1")
    cmp = op.compare_jja(ersst, oisst, lat, a.start_year, a.end_year, landmask=landmask, clim=tuple(a.clim))
    print(op.summary_table(cmp))
    plot_obs.plot_product_comparison(cmp, lat, lon, paths.FIGURES_DIR / "diagnostics" / "obs_products_compare.png",
                                     landmask=landmask)


if __name__ == "__main__":
    main()
