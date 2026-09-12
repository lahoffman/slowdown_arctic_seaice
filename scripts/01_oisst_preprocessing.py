"""
01_oisst_preprocessing.py — OISST v2.1 (AVHRR-only): download, monthly means, regrid.

Stage 1  download each month's daily files from NCEI, average to a monthly mean
         of sst and ice on the 0.25° grid, save OISST_MONTHLY_DIR/oisst_v2.1_mon_YYYYMM.nc,
         delete the dailies (--keep-daily to keep). Resumable: months already
         present are skipped.
Stage 2  area-weighted block average of the monthly fields to the CESM2-LE
         192 x 288 grid → OISST_REGRIDDED (same layout as the ERSST file).

Usage:
  python scripts/01_oisst_preprocessing.py                          # 1990–2025, both stages
  python scripts/01_oisst_preprocessing.py --start-year 2015 --end-year 2016 --workers 6
  python scripts/01_oisst_preprocessing.py --regrid-only            # stage 2 on what is downloaded
  nohup python -u scripts/01_oisst_preprocessing.py > $SLOWDOWN_DATA_ROOT/results/logs/oisst_download.out 2>&1 &
"""

import argparse
import sys
import time
from pathlib import Path

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.observations.oisst import download as dl
from src.data.observations.oisst import regrid_to_cesm2le as rg


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=1990)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--workers", type=int, default=4, help="parallel daily downloads per month")
    p.add_argument("--keep-daily", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--regrid-only", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    print("01  —  OISST v2.1 preprocessing")
    print(f"  years   : {a.start_year}–{a.end_year}")
    print(f"  monthly : {paths.OISST_MONTHLY_DIR}")
    print(f"  regrid  : {paths.OISST_REGRIDDED}")

    if not a.regrid_only:
        t0 = time.time(); n_new = 0
        for y in range(a.start_year, a.end_year + 1):
            for m in range(1, 13):
                f = dl.monthly_file(paths.OISST_MONTHLY_DIR, y, m)
                if f.exists():
                    continue
                out = dl.build_month(y, m, paths.OISST_RAW_DIR, paths.OISST_MONTHLY_DIR,
                                     keep_daily=a.keep_daily, workers=a.workers)
                n_new += out is not None
                print(f"  {y}-{m:02d}  {'ok' if out else 'no data'}   ({time.time() - t0:6.0f} s)", flush=True)
        print(f"  {n_new} new monthly files")

    if not a.download_only:
        with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
            lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
        ds = dl.open_monthly(paths.OISST_MONTHLY_DIR, a.start_year, a.end_year)
        print(f"  regridding {ds.sizes['time']} months ...", flush=True)
        out = rg.regrid_monthly(ds, lat, lon)
        rg.save_regridded(out, paths.OISST_REGRIDDED)


if __name__ == "__main__":
    main()
