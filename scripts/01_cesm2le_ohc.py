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
  python scripts/01_cesm2le_ohc.py --depth 100 300                              # cmip6 group, both depths (hours; network-bound)
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
    p.add_argument("--group", nargs="+", default=["cmip6"], choices=["cmip6", "smbb"],
                   help="forcing group(s); the AWS store has no smbb historical TEMP, so default is cmip6 only")
    p.add_argument("--members", type=int, nargs="+", default=None, help="member indices within the group (default all 50)")
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2100)
    p.add_argument("--workers", type=int, default=4, help="dask threads for the S3 reads (network-bound)")
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
        print("chunks (member, time, z_t, nlat, nlon):", tuple(c[0] for c in ds["TEMP"].chunks), "... per dim")
        print("variables in store:", list(ds.variables))
        dz, zwb = O.layer_geometry(ds["z_t"].values)
        print("z_w_bot [m] top 35:", np.round(zwb[:35] / 100, 1))
        print("levels for 100 / 300 / 700 m:", [O.levels_to(d, zwb) for d in (100, 300, 700)])
        return

    import dask
    dask.config.set(scheduler="threads", num_workers=a.workers)
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
        dz, zwb = O.layer_geometry(ref["z_t"].values)                     # store carries only z_t
        members = list(ref["member_id"].values)
        sel = a.members if a.members is not None else range(len(members))
        with nc.Dataset(paths.CESM2LE_CICE_GRID_FILE) as g:                # CICE grid == POP T-grid
            tlat, tlon = np.array(g["TLAT"][:], float), np.array(g["TLON"][:], float)
        if tlat.shape != (ref.sizes["nlat"], ref.sizes["nlon"]):
            sys.exit(f"grid mismatch: CICE {tlat.shape} vs store {(ref.sizes['nlat'], ref.sizes['nlon'])}")
        idx = O.pop_to_cam_indices(tlat, tlon, lat, lon)
        if ds_h is None:
            print(f"  [note] no historical store for {forcing}: years before 2015 will be NaN")
        nlev = {d: O.levels_to(d, zwb) for d in a.depth}
        for d, n in nlev.items():
            print(f"  OHC{d}: integrating the surface to {zwb[n-1]/100:.0f} m (top {n} levels); the other {zwb.size - n} levels are discarded")
        arrays, ids = {d: [] for d in a.depth}, []
        for m in sel:
            mid = str(members[m]); files = {d: cache / f"ohc{d}_{forcing}_{mid}.npy" for d in a.depth}
            todo = {d: nlev[d] for d in a.depth if not files[d].exists()}
            if todo:
                pop = O.member_series(ds_h, ds_s, mid, todo, dz, years)           # one pass, all missing depths
                for d, arr in pop.items():
                    np.save(files[d], O.regrid(arr, idx, (lat.size, lon.size), landmask))
            for d in a.depth:
                arrays[d].append(np.load(files[d]))
            ids.append(mid)
            print(f"    {forcing} member {m:2d} {mid}: done ({', '.join(f'OHC{d}' for d in a.depth)})", flush=True)
        for d in a.depth:
            out = xr.Dataset({"ohc": (("nens", "nyear", "lat", "lon"), np.stack(arrays[d]))},
                             coords={"nyear": years, "lat": lat, "lon": lon, "member_id": ("nens", ids)})
            out["ohc"].attrs.update(units="J m-2", long_name=f"ocean heat content 0-{d} m, annual mean",
                                    reference_temperature_C=O.T_REF_C, rho=O.RHO, cp=O.CP, n_levels=nlev[d])
            out.attrs.update(source="ncar-cesm2-lens Zarr (AWS)", forcing_variant=forcing,
                             group=O.GROUP_OF_FORCING[forcing], note="POP T-grid → CAM grid, nearest neighbour; land NaN")
            fn = OUT_DIR / f"ohc{d}_cesmle_{O.GROUP_OF_FORCING[forcing]}members_{years[0]}-{years[-1]}.nc"
            out.to_netcdf(fn, encoding={"ohc": {"zlib": True, "complevel": 4}}); print(f"  → {fn}")


if __name__ == "__main__":
    main()
