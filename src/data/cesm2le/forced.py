"""
CESM2-LE forced response fields.

Compute, save, and load the ensemble-mean SST fields that represent the
externally forced climate response.  These are subtracted from individual
ensemble members (during training) or from observations (during inference)
to isolate internal variability.

Currently provides:
  - Ensemble-mean JJA SST on the full 2D atmospheric grid (192 × 288).
    Used by ``src.data.observations.ersst.test_cnn`` to remove the forced
    signal from ERSSTv5 before CNN inference.

The computation mirrors what ``src.cnn.splits.load_jja_sst_demeaned`` does
on the fly (``np.nanmean(sst_jja, axis=0)``), but persists the result so
the observation pipeline can reuse it without reloading all 100 members.

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from typing import List, Tuple


def _load_month_sst_all_members(
    sst_dir: Path,
    month_label: str,
    member_groups: List[str] = ['first50', 'last50'],
) -> np.ndarray:
    """
    Load the SST field for all ensemble members for a single calendar month.

    Reads the pre-separated monthly files:
        sst_cesmle_{group}members_mon_{MONTH}_199001-210012.nc

    Parameters
    ----------
    sst_dir : Path
        Directory containing the monthly SST files.
    month_label : str
        Three-letter calendar month, e.g. ``'JUN'``.
    member_groups : list of str
        Group names to concatenate (default: ``['first50', 'last50']``).

    Returns
    -------
    sst : np.ndarray, shape (nens, nyears, nlat, nlon)
    """
    sst_dir = Path(sst_dir)
    arrays = []
    for group in member_groups:
        fpath = sst_dir / f'sst_cesmle_{group}members_mon_{month_label}_199001-210012.nc'
        if not fpath.exists():
            raise FileNotFoundError(f'Monthly SST file not found: {fpath}')
        ds = nc.Dataset(str(fpath), 'r')
        ds.set_auto_mask(False)
        data = ds.variables['sst_mon'][:].astype(np.float32)
        ds.close()
        arrays.append(data)
    return np.concatenate(arrays, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Compute
# ─────────────────────────────────────────────────────────────────────────────

def compute_ensmean_jja_sst(
    sst_dir: str,
    years: np.ndarray,
    member_groups: List[str] = ['first50', 'last50'],
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the CESM2-LE ensemble-mean JJA SST on the full 2D grid.

    This is the spatial forced response used during CNN training (subtracted
    from each member to isolate internal variability).  Saving it separately
    lets the observation pipeline subtract the same forced field from ERSSTv5.

    Steps
    -----
    1. For each JJA month, load all 100 members → (nens, nyear, nlat, nlon)
    2. Average the 3 months to get JJA mean     → (nens, nyear, nlat, nlon)
    3. Average across ensemble members           → (nyear, nlat, nlon)

    Parameters
    ----------
    sst_dir : str or Path
        Directory containing monthly CESM2 SST files:
        ``sst_cesmle_{group}members_mon_{MONTH}_199001-210012.nc``
    years : np.ndarray
        Year labels, shape (nyear,).
    member_groups : list of str, optional
        Ensemble-member groups to concatenate.
    verbose : bool, optional
        Print progress.

    Returns
    -------
    ensmean_jja : np.ndarray, shape (nyear, nlat, nlon)
        Ensemble-mean JJA SST (the forced component).
    """
    sst_dir = Path(sst_dir)
    jja_months = ['JUN', 'JUL', 'AUG']

    months_data = []
    for m_label in jja_months:
        if verbose:
            print(f'  Ensemble-mean JJA SST: loading {m_label} ...', end='\r')
        sst_m = _load_month_sst_all_members(sst_dir, m_label, member_groups)
        # sst_m: (nens, nyears, nlat, nlon)
        months_data.append(sst_m)

    if verbose:
        print('  Ensemble-mean JJA SST: all months loaded.      ')

    # JJA mean across the 3 months → (nens, nyear, nlat, nlon)
    sst_jja = np.nanmean(months_data, axis=0)

    # Ensemble mean → (nyear, nlat, nlon)
    ensmean_jja = np.nanmean(sst_jja, axis=0)

    if verbose:
        print(f'  Ensemble-mean JJA SST shape: {ensmean_jja.shape}')

    return ensmean_jja


# ─────────────────────────────────────────────────────────────────────────────
# Save / Load
# ─────────────────────────────────────────────────────────────────────────────

def save_ensmean_jja_sst(
    ensmean_jja: np.ndarray,
    years: np.ndarray,
    output_file: str,
) -> None:
    """
    Save the CESM2-LE ensemble-mean JJA SST field to NetCDF.

    Parameters
    ----------
    ensmean_jja : np.ndarray, shape (nyear, nlat, nlon)
        Ensemble-mean JJA SST.
    years : np.ndarray, shape (nyear,)
        Year coordinate.
    output_file : str or Path
        Output NetCDF path.
    """
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    nyear, nlat, nlon = ensmean_jja.shape

    ds = xr.Dataset(
        {
            "sst_ensmean_jja": (("year", "nlat", "nlon"),
                                ensmean_jja.astype(np.float32)),
        },
        coords={
            "year": years,
            "nlat": np.arange(nlat),
            "nlon": np.arange(nlon),
        },
    )
    ds.attrs["description"] = (
        "CESM2-LE ensemble-mean JJA SST (the forced response).  "
        "Mean over all 100 ensemble members of the June-July-August "
        "seasonal-mean SST on the CESM2-LE atmospheric grid (192 x 288)."
    )
    ds["sst_ensmean_jja"].attrs["long_name"] = "Ensemble-mean JJA SST"
    ds["sst_ensmean_jja"].attrs["units"] = "degC"

    encoding = {"sst_ensmean_jja": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(str(output_file), format="NETCDF4", encoding=encoding)
    print(f"✓ Saved ensemble-mean JJA SST to: {output_file}")


def load_ensmean_jja_sst(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a previously saved CESM2-LE ensemble-mean JJA SST field.

    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF written by ``save_ensmean_jja_sst``.

    Returns
    -------
    ensmean : np.ndarray, shape (nyear, nlat, nlon)
        Ensemble-mean JJA SST.
    years : np.ndarray, shape (nyear,)
        Year coordinate.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Ensemble-mean JJA SST file not found: {filepath}"
        )

    with xr.open_dataset(filepath) as ds:
        ensmean = ds["sst_ensmean_jja"].values
        years = ds["year"].values
    return ensmean, years
