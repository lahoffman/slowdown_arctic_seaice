"""
02_oisst_regrid_check.py — did the OISST block-average regridding do what it should?

Draws one month on the native 0.25° grid and on the CESM2-LE grid side by side
(SST global, ice fraction Arctic), plus zonal means of both, which must coincide.

Usage:
  python scripts/02_oisst_regrid_check.py                 # July 2012
  python scripts/02_oisst_regrid_check.py --year 2016 --month 9
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
from src.data.observations.oisst.download import monthly_file
from src.plotting import observations as plot_obs, style as st


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", type=int, default=2012); p.add_argument("--month", type=int, default=7)
    a = p.parse_args()
    st.paper_rc()
    with xr.open_dataset(monthly_file(paths.OISST_MONTHLY_DIR, a.year, a.month)) as ds:
        native = dict(sst=ds["sst"].values, ice=ds["ice"].values, lat=ds["lat"].values, lon=ds["lon"].values,
                      label=f"native 0.25°, {a.year}-{a.month:02d}")
    with xr.open_dataset(paths.OISST_REGRIDDED) as ds:
        t = ds["time"].dt
        i = int(np.where((t.year.values == a.year) & (t.month.values == a.month))[0][0])
        regridded = dict(sst=ds["sst_obs"].values[i], ice=ds["ice_obs"].values[i],
                         lat=ds["lat_cesm2"].values[:, 0], lon=ds["lon_cesm2"].values[0, :],
                         label="CESM2-LE grid (block average)")
    landmask = None
    if paths.LANDMASK_FILE.exists():
        with nc.Dataset(paths.LANDMASK_FILE) as d:
            landmask = np.array(d["landmask"][:])
    w_n = np.cos(np.deg2rad(native["lat"]))[:, None]; w_r = np.cos(np.deg2rad(regridded["lat"]))[:, None]
    m = lambda f, w: np.nansum(f * w) / np.nansum(np.isfinite(f) * w)
    print(f"  area-mean SST  native {m(native['sst'], w_n):.3f}  regridded {m(regridded['sst'], w_r):.3f} °C")
    print(f"  area-mean ice  native {m(native['ice'], w_n):.4f}  regridded {m(regridded['ice'], w_r):.4f}")
    plot_obs.plot_regrid_check(native, regridded, paths.FIGURES_DIR / "diagnostics" / "oisst_regrid_check.png",
                               landmask=landmask)


if __name__ == "__main__":
    main()
