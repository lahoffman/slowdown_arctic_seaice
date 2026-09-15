#!/usr/bin/env python3
"""
01_cesm2le_ohc.py — annual upper-ocean heat content for CESM2-LE from the AWS Zarr store (no TEMP download).

Per forcing group (cmip6 → members 0–49, smbb → 50–99) and depth, writes
  DATA_ROOT/cesm2le/ohc/ohc<DEPTH>_cesmle_<group>members_1990-2100.nc     variable ohc (nens, nyear, lat, lon), J m^-2
on the CAM grid, with member_id recorded so the ordering can be checked against the SST/SIE files.
Resumable: members already in the per-member cache (…/ohc/cache/) are skipped.

Usage:
  python scripts/01_cesm2le_ohc.py --list                     # can we see the store? which TEMP entries, chunking?
  python scripts/01_cesm2le_ohc.py --depth 100 --group cmip6 --members 0 1      # smoke test, two members
  python scripts/01_cesm2le_ohc.py --depth 100 300 --group cmip6 smbb           # everything (hours; network-bound)
Needs: pip install intake-esm s3fs zarr dask
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
from src.data.cesm2le import ohc as O

OUT_DIR = paths.CESM2LE_DIR / "ohc"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print the catalog's TEMP entries and one store's chunking, then exit")
    p.add_argument("--depth", type=int, nargs="+", default=[100], help="integration depths in m (100 300 700)")
    p.add_argument("--group", nargs="+", default=["cmip6", "smbb"], choices=["cmip6", "smbb"])
    p.add_argument("--members", type=int, nargs="+", default=None, help="member indices within the group (default all 50)")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2100)
    p.add_argument("--catalog", default=O.CATALOG)
    return p.parse_args()


def stores_for(df, forcing: str):
    """(historical, ssp370) store paths for one forcing variant; None where absent."""
    sub = df[df["forcing_variant"].str.lower() == forcing] if "forcing_variant" in df else df
    def pick(exp):
        m = sub[sub["experiment"].str.lower().str.contains(exp)]
        return m["path"].iloc[0] if len(m) else None
    return pick("hist"), pick("ssp")


def main():
    a = parse_args()
    df = O.catalog_temp_entries(a.catalog)
    if df.empty:
        sys.exit("No ocean monthly TEMP in the catalog — fall back to OPeNDAP hyperslabs (docs/OHC_PLAN.md, route 2).")
    cols = [c for c in ("variable", "component", "experiment", "forcing_variant", "frequency", "start_time", "end_time", "path") if c in df]
    print(df[cols].to_string())
    if a.list:
        ds = O.open_store(df["path"].iloc[0])
        print("\nstore:", df["path"].iloc[0]); print(ds["TEMP"])
        print("chunks (member, time, z_t, nlat, nlon):", ds["TEMP"].chunks)
        print("z_w_bot [m] top 35:", np.round(ds["z_w_bot"].values[:35] / 100, 1))
        return

    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        landmask = np.array(d["landmask"][:])
    years = np.arange(a.start_year, a.end_year + 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True); cache = OUT_DIR / "cache"; cache.mkdir(exist_ok=True)

    for forcing in a.group:
        hist, ssp = stores_for(df, forcing)
        print(f"\n== {forcing}: historical={hist}\n           ssp370={ssp}")
        ds_h = O.open_store(hist) if hist else None
        ds_s = O.open_store(ssp) if ssp else None
        ref = ds_h or ds_s
        dz, zwb = ref["dz"].values, ref["z_w_bot"].values
        members = list(ref["member_id"].values)
        sel = a.members if a.members is not None else range(len(members))
        idx = O.pop_to_cam_indices(ref["TLAT"].values, ref["TLONG"].values, lat, lon)
        for depth in a.depth:
            nlev = O.levels_to(depth, zwb)
            print(f"  depth {depth} m → top {nlev} levels (bottom {zwb[nlev-1]/100:.0f} m)")
            arrays, ids = [], []
            for m in sel:
                mid = str(members[m]); f = cache / f"ohc{depth}_{forcing}_{mid}.npy"
                if f.exists():
                    arr = np.load(f)
                else:
                    pop = O.member_series(ds_h, ds_s, mid, nlev, dz, years)          # (nyear, nlat, nlon), computes
                    arr = O.regrid(pop, idx, (lat.size, lon.size), landmask)
                    np.save(f, arr)
                arrays.append(arr); ids.append(mid)
                print(f"    {forcing} member {m:2d} {mid}: done", flush=True)
            out = xr.Dataset({"ohc": (("nens", "nyear", "lat", "lon"), np.stack(arrays))},
                             coords={"nyear": years, "lat": lat, "lon": lon, "member_id": ("nens", ids)})
            out["ohc"].attrs.update(units="J m-2", long_name=f"ocean heat content 0-{depth} m, annual mean",
                                    reference_temperature_C=O.T_REF_C, rho=O.RHO, cp=O.CP, n_levels=nlev)
            out.attrs.update(source="ncar-cesm2-lens Zarr (AWS)", forcing_variant=forcing,
                             group=O.GROUP_OF_FORCING[forcing], note="POP T-grid → CAM grid, nearest neighbour; land NaN")
            fn = OUT_DIR / f"ohc{depth}_cesmle_{O.GROUP_OF_FORCING[forcing]}members_{years[0]}-{years[-1]}.nc"
            out.to_netcdf(fn, encoding={"ohc": {"zlib": True, "complevel": 4}}); print(f"  → {fn}")


if __name__ == "__main__":
    main()
