# Ocean heat content from CESM2-LE without downloading the ocean (2026-09-15)

Why: (a) the GMT test that would actually bear on Labe & Barnes (2022) needs their predictor — OHC100 maps — not the
GMT state; (b) the sea-ice ledger's deferred item 8.5 (Barents–Kara / Atlantic-inflow heat as the missing decadal memory)
needs OHC0–300 north of 50°N. Both need only the top ~10–30 of POP's 60 levels, annual means, 100 members, 1990–2100.

## The data problem
Full `pop.h.TEMP` monthly files: 60 levels × 384 × 320 × 120 months × 4 B ≈ 3.5 GB per member-decade; 100 members ×
11 decades ≈ 4 TB. Not an option through `download.py` (whole-file HTTPS from GDEX). Two ways to fetch only the top levels:

1. **AWS CESM2-LENS Zarr (preferred if TEMP is there).** Bucket `s3://ncar-cesm2-lens/` (anonymous, us-west-2), intake-esm
   catalog `https://ncar-cesm2-lens.s3.amazonaws.com/catalogs/aws-cesm2-le.json`. Zarr is chunked, so slicing `z_t[:10]` transfers
   only those chunks. Check first whether ocean monthly TEMP is included (SST certainly is):
   ```python
   import intake
   cat = intake.open_esm_datastore("https://ncar-cesm2-lens.s3.amazonaws.com/catalogs/aws-cesm2-le.json")
   print(cat.df.query("component == 'ocn'")[["variable", "frequency", "experiment", "path"]].drop_duplicates("variable"))
   ```
   If `TEMP` is listed:
   ```python
   import xarray as xr, s3fs
   fs = s3fs.S3FileSystem(anon=True)
   ds = xr.open_zarr(fs.get_mapper(path), consolidated=True)            # path from cat.df
   top = ds["TEMP"].isel(z_t=slice(0, 10))                                # 0–100 m (10 × 10 m layers); 30 for 0–300 m
   ann = top.resample(time="YS").mean()                                   # annual means, lazy
   ann.to_netcdf(...)                                                     # per member or in member chunks
   ```
   Volume for 10 levels × annual: 100 × 111 yr × 10 × 384 × 320 × 4 B ≈ 55 GB before regridding; do the vertical
   integral (`dz`-weighted, TEMP in °C → J m⁻²) *before* writing and it is 5.5 GB per depth range.
2. **OPeNDAP on the NCAR THREDDS (if 1 fails).** GDEX/RDA serve d651056 through THREDDS; an OPeNDAP URL of the form
   `https://thredds.rda.ucar.edu/thredds/dodsC/files/g/d651056/CESM2-LE/ocn/proc/tseries/month_1/TEMP/<file>.nc`
   accepts hyperslabs, so `xr.open_dataset(url)["TEMP"].isel(z_t=slice(0, 10))` transfers only the requested levels.
   Slower and per-file, but works with the existing member/decade file naming in `download.py`. Verify the path exists
   with one file before scripting.

## Computing OHC correctly (note on the old snippet)
POP gx1v7 has 10-m layers in the upper 150 m (`z_t` centres at 5, 15, … m; `dz` in the file). The snippet in the notes
used `TEMP[:, :3]` (0–30 m) and multiplied by h = 100 m — that is a 30-m temperature scaled as if it were 100 m of water.
The integral is
```
OHC_0-H = ρ c_p Σ_k (T_k − T_ref) dz_k   over levels with z_w_bot ≤ H,   ρ = 1026 kg m⁻³, c_p = 3990 J kg⁻¹ K⁻¹
```
(T_ref cancels in anomalies; use levels 0–9 for 100 m, 0–29 for ~300 m — check `z_w_bot`). Then regrid to the CAM grid
with the same displaced-pole → lat/lon nearest-neighbour used for `aice` (`src/data/cesm2le/regrid.py`), demean per
forcing group, and store annual `ohc100` / `ohc300` as `(nens, nyear, nx, ny)` next to the SST inputs.

## What we would run with it
- **GMT/LB22 test (their predictor, our controls):** ridge/logistic on OHC100 PCs and a small ANN as in LB22 → GMT
  slowdown labels, onset-inclusive vs offset; GMT-state baseline beside it; `r(OHC100 projection, GMT(t))` — the number
  that closes the "how much of their skill is the coupling" question. Only then can §9 say more than "in principle".
- **SIE ledger (8.5):** OHC0–300 Barents–Kara / Nordic / Labrador indices at onset → offset trend anomaly residual after
  SIE (+ volume); pointwise correlation maps; if the Atlantic-inflow heat adds ≳ 0.03 it is the first ocean predictor
  that beats the IPO and the paper's ledger gets a fourth line.
Both are `09_*` scripts on scalar/PC inputs — no CNN needed for the first pass.

## Cost estimate
Catalog check: minutes. Zarr extraction of 10 levels, annual, 100 members: hours of I/O on profx (network-bound), one
script. Regrid + demean: an hour. The two analyses: an afternoon each. Decide after the catalog check whether to do it
inside this paper (it would make §9 a result instead of a bound) or as the follow-up.

## Code (2026-09-15 night)
`src/data/cesm2le/ohc.py` (catalog lookup, lazy store open, dz-weighted column integral, annual mean, POP→CAM nearest map)
and `scripts/01_cesm2le_ohc.py`. Decision: **OHC goes in this paper**, OHC100 first (LB22's predictor: "vertical heat content
integral 0–100 m", annual mean — confirmed from their §2), OHC300 for the Barents–Kara ledger item.

Access check (profx, arcticwatch env):
```
pip install intake-esm s3fs zarr dask
python scripts/01_cesm2le_ohc.py --list
```
→ prints the catalog rows for ocean monthly TEMP (or exits saying there are none → OPeNDAP route), the TEMP array with
its chunking, and z_w_bot. If chunks span all 60 levels, the slice still costs full-column transfer per chunk; the script
then simply computes per member (one member's top levels at a time) — no special reader/writer, dask handles it.
Smoke test: `--depth 100 --group cmip6 --members 0 1`. Full: `--depth 100 300 --group cmip6 smbb` (resumable via cache/).
Member ordering: `member_id` is stored in the output; verify against the SST files by correlating one member's global-mean
OHC100 anomaly with its global-mean SST anomaly before using the arrays together.
