"""
OISST v2.1 daily files → monthly means on the native 0.25° grid.

NCEI serves one NetCDF per day (~1.6 MB):
  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/YYYYMM/oisst-avhrr-v02r01.YYYYMMDD.nc
with variables sst, anom, err, ice (time, zlev, lat, lon). We download a month at a
time, average to a monthly mean of sst and ice, save that, and delete the dailies.
"""

from __future__ import annotations

import calendar
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import numpy as np
import xarray as xr

BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"


def daily_url(year: int, month: int, day: int) -> str:
    return f"{BASE_URL}/{year:04d}{month:02d}/oisst-avhrr-v02r01.{year:04d}{month:02d}{day:02d}.nc"


def monthly_file(monthly_dir: Path, year: int, month: int) -> Path:
    return Path(monthly_dir) / f"oisst_v2.1_mon_{year:04d}{month:02d}.nc"


def _fetch(url: str, dest: Path, retries: int = 4, timeout: int = 120) -> Optional[Path]:
    """Download one file with retries; returns None if it does not exist upstream (404)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            return dest
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** k)
        except Exception:
            time.sleep(2 ** k)
    raise RuntimeError(f"failed after {retries} tries: {url}")


def download_month(year: int, month: int, raw_dir: Path, workers: int = 4) -> List[Path]:
    """All daily files of one month (missing days are skipped)."""
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    days = range(1, calendar.monthrange(year, month)[1] + 1)
    jobs = [(daily_url(year, month, d), raw_dir / f"oisst-avhrr-v02r01.{year:04d}{month:02d}{d:02d}.nc") for d in days]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda j: _fetch(*j), jobs))
    return [p for p in out if p is not None]


def monthly_mean(daily_files: List[Path]) -> xr.Dataset:
    """Monthly mean sst (°C) and ice fraction from the daily files (running sums; no dask needed)."""
    sum_sst = sum_ice = n_sst = n_ice = None
    for f in daily_files:
        with xr.open_dataset(f) as ds:
            d = ds.squeeze(drop=True)                        # drop time=1, zlev=1
            sst, ice = d["sst"].values.astype(np.float64), d["ice"].values.astype(np.float64)
            lat, lon = d["lat"].values, d["lon"].values
        if sum_sst is None:
            sum_sst, sum_ice = np.zeros_like(sst), np.zeros_like(ice)
            n_sst, n_ice = np.zeros(sst.shape, np.int16), np.zeros(ice.shape, np.int16)
        ok = np.isfinite(sst); sum_sst[ok] += sst[ok]; n_sst += ok
        ok = np.isfinite(ice); sum_ice[ok] += ice[ok]; n_ice += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        sst_m = np.where(n_sst > 0, sum_sst / n_sst, np.nan).astype("f4")
        ice_m = np.where(n_ice > 0, sum_ice / n_ice, 0.0).astype("f4")     # no report = ice-free
    out = xr.Dataset({"sst": (("lat", "lon"), sst_m), "ice": (("lat", "lon"), ice_m),
                      "ndays": np.int16(len(daily_files))},
                     coords={"lat": lat, "lon": lon})
    out["sst"].attrs.update(units="degC", long_name="monthly mean OISST v2.1 SST")
    out["ice"].attrs.update(units="1", long_name="monthly mean sea-ice concentration (0 where ice-free)")
    out.attrs.update(source="NOAA OISST v2.1 AVHRR-only daily", url=BASE_URL)
    return out


def build_month(year: int, month: int, raw_dir: Path, monthly_dir: Path, keep_daily: bool = False,
                workers: int = 4) -> Optional[Path]:
    """Download → monthly mean → save; returns the monthly file (None if no days available)."""
    target = monthly_file(monthly_dir, year, month)
    if target.exists():
        return target
    files = download_month(year, month, raw_dir, workers)
    if not files:
        return None
    mm = monthly_mean(files)
    target.parent.mkdir(parents=True, exist_ok=True)
    mm.to_netcdf(target, encoding={v: {"zlib": True, "complevel": 4} for v in ("sst", "ice")})
    if not keep_daily:
        for f in files:
            f.unlink(missing_ok=True)
    return target


def open_monthly(monthly_dir: Path, start_year: int, end_year: int) -> xr.Dataset:
    """Concatenate the monthly files into one Dataset with a ``time`` axis (month starts)."""
    files, times = [], []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            f = monthly_file(monthly_dir, y, m)
            if f.exists():
                files.append(f); times.append(np.datetime64(f"{y:04d}-{m:02d}-01"))
    if not files:
        raise FileNotFoundError(f"no monthly OISST files in {monthly_dir}")
    ds = xr.concat([xr.open_dataset(f) for f in files], dim="time")
    return ds.assign_coords(time=np.array(times))
