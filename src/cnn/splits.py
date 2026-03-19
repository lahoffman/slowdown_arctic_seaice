"""
Train / Validate / Test data preparation for the JJA SST CNN.

Pipeline overview
-----------------
1. load_jja_sst_demeaned   — load monthly SST (JUN/JUL/AUG), compute seasonal
                             mean, subtract ensemble mean  →  (nens, nyear, nx, ny)
2. block_tvt_split         — split ensemble members into T/V/T blocks and flatten
                             the block × time dimensions into a single sample axis
3. standardize             — compute global mean/std from ocean-only training pixels
                             and apply to all three splits
4. apply_landmask          — fill land pixels with a sentinel value (-10)
5. save_tvt_split          — persist one split's arrays + norm stats to NetCDF
6. load_tvt_split          — reload a saved split for downstream use (e.g. XAI)

Array conventions
-----------------
After splitting and flattening the dimensions are:
    SST arrays      : (n_samples, nx, ny)   — one map per ensemble-year sample
    slowdown arrays : (n_samples,)           — binary label per sample

Ensemble blocks
---------------
100 ensemble members are arranged as 10 blocks of 10 members each.
For each of the 9 train/val/test splits:
    - test  block  : 1 block  (10 members × nyear samples)
    - val   block  : 1 block  (next block, wrapping)
    - train blocks : 8 blocks (80 members × nyear samples)

Authors: Lauren Hoffman  <lhoffma2@ucsc.edu>
"""

import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path
from typing import Iterator, Optional, Tuple, Dict, List


# =============================================================================
# 1. Load and prepare JJA SST
# =============================================================================

def load_jja_sst_demeaned(
    sst_monthly_template: Dict[str, str],
    member_groups: List[str] = ['first50', 'last50'],
    start_year: int = 1990,
    end_year: int = 2040,
    file_years: str = '199001-210012',
    sst_varname: str = 'sst_mon',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load monthly SST for June, July, August; compute the JJA seasonal mean;
    subtract the ensemble mean to isolate internal variability.

    Reads one NetCDF per calendar month per member group (produced by
    scripts/01_cesm2le_preprocessing.py via src/data/cesm2le/combine.py and
    separate_by_month), concatenates groups along the ensemble axis, then
    returns the ensemble-demeaned JJA mean over the requested year range.

    Parameters
    ----------
    sst_monthly_template : dict
        Mapping {group_label: path_template_string} where the template
        contains ``{month}`` as a placeholder.  Use the ``CESM2LE_SST_MONTHLY``
        dict from configs.paths, e.g.:
            ``{'first50': '.../sst_cesmle_first50members_mon_{month}_199001-210012.nc'}``
    member_groups : list of str
        Keys into ``sst_monthly_template`` to load and concatenate, in the
        order that sets the ensemble-member indexing (default: first50 then last50).
    start_year : int
        First year to include in the output (inclusive, default: 1990).
        The SST year matches the September SIE target year — 
        JJA of year *t* predicts September of year *t*.
    end_year : int
        Last year to include in the output (inclusive, default: 2040).
    file_years : str
        Year-range suffix in the source filenames, e.g. ``'199001-210012'``.
        Used only to construct the year index array for slicing.
    sst_varname : str
        Name of the SST variable inside the NetCDF files (default: ``'sst_mon'``).

    Returns
    -------
    sst_jja_demeaned : np.ndarray
        Ensemble-mean-removed JJA SST, shape ``(nens, nyear, nx, ny)``.
    years : np.ndarray
        Year labels, shape ``(nyear,)``, e.g. ``array([1990, ..., 2040])``.
    """

    jja_months = ['JUN', 'JUL', 'AUG']

    # Build full year array from file start year
    file_start_year = int(file_years[:4])
    file_end_year   = int(file_years[5:9])
    all_years       = np.arange(file_start_year, file_end_year + 1)

    # Year slice indices within the full file
    idx_start = int(np.where(all_years == start_year)[0][0])
    idx_end   = int(np.where(all_years == end_year)[0][0]) + 1   # exclusive

    months_data = []
    for month in jja_months:
        group_arrays = []
        for group in member_groups:
            fpath = sst_monthly_template[group].format(month=month)
            if not Path(fpath).exists():
                raise FileNotFoundError(
                    f"SST file not found for group='{group}', month='{month}':\n  {fpath}"
                )
            with nc.Dataset(fpath, 'r') as ds:
                arr = ds.variables[sst_varname][:]   # (nens_group, ntime, nx, ny)
                group_arrays.append(np.array(arr, dtype=np.float32))

        sst_month = np.concatenate(group_arrays, axis=0)        # (nens, ntime, nx, ny)
        months_data.append(sst_month[:, idx_start:idx_end, :, :])  # slice years (nmon, nens, ntime, nx, ny)

    # JJA mean across the 3 months (axis=0 of the list, not the array axis)
    sst_jja = np.nanmean(months_data, axis=0)   # (nens, nyear, nx, ny)

    # Subtract ensemble mean (removes forced response)
    ens_mean    = np.nanmean(sst_jja, axis=0, keepdims=True)  # (1, nyear, nx, ny)
    sst_jja_dem = sst_jja - ens_mean

    years = all_years[idx_start:idx_end]

    print(f"Loaded JJA SST: shape {sst_jja_dem.shape}, years {years[0]}-{years[-1]}")
    return sst_jja_dem, years


# =============================================================================
# 2. Block-based train / validate / test splitting
# =============================================================================

def _get_block_indices(n_splits: int = 9, n_blocks: int = 10):
    """
    Generate test / val / train block indices for each of the n_splits splits.

    """
    for k in range(n_splits):
        test_block   = k
        val_block    = k + 1
        train_blocks = [i for i in range(n_blocks)
                        if i != test_block and i != val_block]
        yield k, test_block, val_block, train_blocks


def block_tvt_split(
    arr: np.ndarray,
    train_blocks: List[int],
    val_block: int,
    test_block: int,
    n_blocks: int = 10,
    block_size: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split an array along its first (ensemble) dimension into T/V/T sets.
    

    The 100-member ensemble is treated as ``n_blocks`` groups of ``block_size``
    members each. The blocks are made to group ensemble members that have
    different forcing in the CESM2-LE, so that T/V/T splits are made along the
    axis of model forcing rather than just general ensemble member.

    The function selects the requested blocks, then flattens
    the block and all subsequent leading dimensions into a single sample axis.

    This works for arrays of any trailing shape, so the same call handles both
    spatial SST data ``(nens, nyear, nx, ny)`` and scalar labels ``(nens, nyear)``.

    Parameters
    ----------
    arr : np.ndarray
        Shape ``(nens, ntime, nlat, nlon)`` where ``nens = n_blocks × block_size``.
    train_blocks : list of int
        Indices of the blocks assigned to training.
    val_block : int
        Index of the validation block.
    test_block : int
        Index of the test block.
    n_blocks : int
        Total number of blocks (default: 10).
    block_size : int
        Number of ensemble members per block (default: 10).

    Returns
    -------
    tr, va, te : np.ndarray
        - ``tr`` : ``(n_train_blocks × block_size × nyear, ...)``
                   e.g. ``(4000, nx, ny)`` for SST or ``(4000,)`` for labels
        - ``va`` : ``(block_size × nyear, ...)``  e.g. ``(500, nx, ny)``
        - ``te`` : ``(block_size × nyear, ...)``  e.g. ``(500, nx, ny)``

    """
    nens = arr.shape[0]
    if nens != n_blocks * block_size:
        raise ValueError(
            f"Expected arr.shape[0] = n_blocks × block_size = "
            f"{n_blocks} × {block_size} = {n_blocks * block_size}, "
            f"but got {nens}."
        )

    suffix_shape = arr.shape[1:]   # (nt, ...) after nens

    # STEP 1: Reshape → (n_blocks, block_size, nt, ...)
    blocked = arr.reshape(n_blocks, block_size, *suffix_shape)

    # STEP 2: Select along axis 1 (the block_size / inner-member axis)
    # Selecting on axis 0 would group across forcing types; axis 1 groups
    # members *within* each forcing block, which is the intended split axis.
    # SST:    (10, 10, 50, nx, ny)[:, [0,2..8], ...] → (10, 8, 50, nx, ny)
    # labels: (10, 10, 50)        [:, [0,2..8], ...] → (10, 8, 50)
    # val/te: scalar index on axis 1 collapses that dim
    # SST:    (10, 10, 50, nx, ny)[:, k, ...]        → (10, 50, nx, ny)
    # labels: (10, 10, 50)        [:, k, ...]        → (10, 50)
    tr_raw = blocked[:, train_blocks, ...]   # (n_blocks, n_tr_selected, nt, ...)
    va_raw = blocked[:, val_block,    ...]   # (n_blocks, nt, ...)
    te_raw = blocked[:, test_block,   ...]   # (n_blocks, nt, ...)

    # STEP 3a: Merge n_blocks × n_selected into one dimension
    # tr: (10, 8, 50, nx, ny) → (80, 50, nx, ny)
    # va: (10, 50, nx, ny)    → already (n_blocks, nt, ...), treated as (10, 50, nx, ny)
    # te: same as va
    n_tr  = len(train_blocks)
    tr_3a = tr_raw.reshape(n_blocks * n_tr, *suffix_shape)   # (80, nt, ...)
    va_3a = va_raw                                            # (10, nt, ...)
    te_3a = te_raw                                            # (10, nt, ...)

    # STEP 3b: Flatten time (and blocks for va/te) into the sample axis
    # tr: (80, 50, nx, ny) → (4000, nx, ny)
    # va: (10, 50, nx, ny) → ( 500, nx, ny)
    # te: (10, 50, nx, ny) → ( 500, nx, ny)
    trailing = suffix_shape[1:] if len(suffix_shape) > 1 else ()
    tr = tr_3a.reshape(-1, *trailing)
    va = va_3a.reshape(-1, *trailing)
    te = te_3a.reshape(-1, *trailing)

    return tr, va, te


def iter_splits(
    sst: np.ndarray,
    slowdown: np.ndarray,
    n_splits: int = 9,
    n_blocks: int = 10,
    block_size: int = 10,
) -> Iterator[Dict]:
    """
    Iterate over all n_splits train/validate/test partitions.

    Yields one dict per split containing the raw (pre-standardisation) SST
    and slowdown arrays for each partition, plus the split metadata.

    Parameters
    ----------
    sst : np.ndarray
        Ensemble-demeaned JJA SST, shape ``(nens, nyear, nx, ny)``.
    slowdown : np.ndarray
        Binary slowdown labels, shape ``(nens, nyear)``.
        Must have the same ``nens`` and ``nyear`` as ``sst``.
    n_splits : int
        Number of TVT partitions to generate (default: 9).
    n_blocks : int
        Number of ensemble blocks (default: 10).
    block_size : int
        Members per block (default: 10).

    Yields
    ------
    dict with keys:
        ``split_idx``   : int
        ``test_block``  : int
        ``val_block``   : int
        ``train_blocks``: list of int
        ``sst_tr``      : np.ndarray  (ntr, nx, ny)
        ``sst_va``      : np.ndarray  (nva, nx, ny)
        ``sst_te``      : np.ndarray  (nte, nx, ny)
        ``slow_tr``     : np.ndarray  (ntr,)
        ``slow_va``     : np.ndarray  (nva,)
        ``slow_te``     : np.ndarray  (nte,)
    """
    for k, test_block, val_block, train_blocks in _get_block_indices(n_splits, n_blocks):
        sst_tr, sst_va, sst_te = block_tvt_split(
            sst, train_blocks, val_block, test_block, n_blocks, block_size
        )
        slow_tr, slow_va, slow_te = block_tvt_split(
            slowdown, train_blocks, val_block, test_block, n_blocks, block_size
        )
        yield {
            'split_idx':   k,
            'test_block':  test_block,
            'val_block':   val_block,
            'train_blocks': train_blocks,
            'sst_tr':  sst_tr,
            'sst_va':  sst_va,
            'sst_te':  sst_te,
            'slow_tr': slow_tr,
            'slow_va': slow_va,
            'slow_te': slow_te,
        }


# =============================================================================
# 3. Standardisation
# =============================================================================

def standardize(
    sst_tr: np.ndarray,
    sst_va: np.ndarray,
    sst_te: np.ndarray,
    landmask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Standardise SST arrays using global statistics computed from ocean-only
    training-data pixels.

    The mean and standard deviation are estimated over all training samples
    and all ocean grid cells together (a single global scalar pair), following
    the "globalSTD" convention used throughout this project.  Only ocean pixels
    (``landmask == 0``) are included in the statistics to avoid contamination
    by the land fill value.

    All three splits are standardised using the *training* statistics so that
    validation and test data are treated as unseen, avoiding data leakage.

    Parameters
    ----------
    sst_tr : np.ndarray
        Training SST, shape ``(ntr, nx, ny)``.
    sst_va : np.ndarray
        Validation SST, shape ``(nva, nx, ny)``.
    sst_te : np.ndarray
        Test SST, shape ``(nte, nx, ny)``.
    landmask : np.ndarray
        Land mask, shape ``(nx, ny)``.  0 = ocean, 1 = land.

    Returns
    -------
    sst_tr_std : np.ndarray  (ntr, nx, ny)
    sst_va_std : np.ndarray  (nva, nx, ny)
    sst_te_std : np.ndarray  (nte, nx, ny)
    mu_train   : float   — global ocean mean of training data
    sigma_train: float   — global ocean std  of training data
    """
    lm = landmask[np.newaxis, :, :]                  # (1, nx, ny) for broadcasting
    sst_tr_ocean = np.where(lm == 0, sst_tr, np.nan)  # mask land → NaN

    mu_train    = float(np.nanmean(sst_tr_ocean))
    sigma_train = float(np.nanstd(sst_tr_ocean))

    if sigma_train == 0:
        raise ValueError("Training SST standard deviation is zero — cannot standardise.")

    sst_tr_std = (sst_tr - mu_train) / sigma_train
    sst_va_std = (sst_va - mu_train) / sigma_train
    sst_te_std = (sst_te - mu_train) / sigma_train

    return sst_tr_std, sst_va_std, sst_te_std, mu_train, sigma_train


# =============================================================================
# 4. Land masking
# =============================================================================

def apply_landmask(
    X: np.ndarray,
    landmask: np.ndarray,
    fill_value: float = -10.0,
) -> np.ndarray:
    """
    Replace land pixels with a sentinel fill value.

    Called *after* standardisation so that the fill value sits outside the
    range of any physically meaningful standardised SST.  The model learns
    to ignore these fixed-value pixels.  Before running LRP (XAI), replace
    the fill value with 0 to avoid spurious attributions (see src/xai/lrp.py).

    Parameters
    ----------
    X : np.ndarray
        Standardised SST, shape ``(n_samples, nx, ny)``.
    landmask : np.ndarray
        Land mask, shape ``(nx, ny)``.  1 = land, 0 = ocean.
    fill_value : float
        Value to assign to land pixels (default: ``-10.0``).

    Returns
    -------
    X_masked : np.ndarray
        Same shape as ``X`` with land pixels set to ``fill_value``.
    """
    lm = landmask[np.newaxis, :, :]    # (1, nx, ny)
    return np.where(lm == 1, fill_value, X)


# =============================================================================
# 5. Save / load TVT splits
# =============================================================================

def save_tvt_split(
    sst_tr: np.ndarray,
    sst_va: np.ndarray,
    sst_te: np.ndarray,
    slow_tr: np.ndarray,
    slow_va: np.ndarray,
    slow_te: np.ndarray,
    mu_train: float,
    sigma_train: float,
    split_idx: int,
    savepath: Path,
    attrs: Optional[dict] = None,
) -> None:
    """
    Save one TVT split (standardised SST + slowdown labels + norm stats) to NetCDF.

    The normalisation statistics ``mu_train`` and ``sigma_train`` are stored
    as scalar data variables so they can be recovered when loading the split for
    downstream XAI without re-running the preprocessing pipeline.

    Parameters
    ----------
    sst_tr, sst_va, sst_te : np.ndarray
        Standardised (and landmasked) SST for each partition.
        Shapes: ``(ntr, nx, ny)``, ``(nva, nx, ny)``, ``(nte, nx, ny)``.
    slow_tr, slow_va, slow_te : np.ndarray
        Binary slowdown labels (0/1), shapes: ``(ntr,)``, ``(nva,)``, ``(nte,)``.
    mu_train : float
        Global ocean mean of the training SST (before standardisation).
    sigma_train : float
        Global ocean std of the training SST (before standardisation).
    split_idx : int
        Integer index of this split (0-based), stored as a file attribute.
    savepath : Path or str
        Output file path.  Parent directories are created if they don't exist.
    attrs : dict, optional
        Additional global attributes to store in the NetCDF file.
    """
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    nx, ny = sst_tr.shape[1], sst_tr.shape[2]

    ds = xr.Dataset(
        {
            "sst_tr":    (("ntr", "nx", "ny"), sst_tr.astype(np.float32)),
            "sst_va":    (("nva", "nx", "ny"), sst_va.astype(np.float32)),
            "sst_te":    (("nte", "nx", "ny"), sst_te.astype(np.float32)),
            "slow_tr":   (("ntr",),            slow_tr.astype(np.int8)),
            "slow_va":   (("nva",),            slow_va.astype(np.int8)),
            "slow_te":   (("nte",),            slow_te.astype(np.int8)),
            "mu_train":  ((), np.float32(mu_train)),
            "sigma_train": ((), np.float32(sigma_train)),
        },
        coords={
            "ntr": np.arange(sst_tr.shape[0]),
            "nva": np.arange(sst_va.shape[0]),
            "nte": np.arange(sst_te.shape[0]),
            "nx":  np.arange(nx),
            "ny":  np.arange(ny),
        },
    )

    ds.attrs["split_idx"]   = split_idx
    ds.attrs["description"] = (
        "JJA SST (standardised, land-masked) + September slowdown labels "
        f"for TVT split {split_idx}.  "
        "Standardisation used ocean-only global mean/std from training data."
    )
    if attrs:
        ds.attrs.update(attrs)

    # Variable-level metadata
    ds["sst_tr"].attrs["long_name"]    = "Training JJA SST (standardised)"
    ds["sst_va"].attrs["long_name"]    = "Validation JJA SST (standardised)"
    ds["sst_te"].attrs["long_name"]    = "Test JJA SST (standardised)"
    ds["slow_tr"].attrs["long_name"]   = "Training slowdown labels (1=slowdown)"
    ds["slow_va"].attrs["long_name"]   = "Validation slowdown labels"
    ds["slow_te"].attrs["long_name"]   = "Test slowdown labels"
    ds["mu_train"].attrs["long_name"]  = "Training ocean mean SST (raw, before standardisation)"
    ds["mu_train"].attrs["units"]      = "degC"
    ds["sigma_train"].attrs["long_name"] = "Training ocean std SST (raw)"
    ds["sigma_train"].attrs["units"]     = "degC"

    encoding = {v: {"zlib": True, "complevel": 4}
                for v in ds.data_vars if ds[v].ndim > 0}
    ds.to_netcdf(savepath, format="NETCDF4", encoding=encoding)
    print(f"  Split {split_idx} saved → {savepath}")


def load_tvt_split(filepath: Path) -> Dict:
    """
    Load a saved TVT split from NetCDF.

    Returns a dict with the same keys as produced by ``save_tvt_split``:
    ``sst_tr``, ``sst_va``, ``sst_te``, ``slow_tr``, ``slow_va``, ``slow_te``,
    ``mu_train``, ``sigma_train``, and ``split_idx``.

    Parameters
    ----------
    filepath : Path or str
        Path to a NetCDF file written by ``save_tvt_split``.

    Returns
    -------
    dict
        All arrays and scalars from the saved split.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"TVT split file not found: {filepath}")

    with xr.open_dataset(filepath) as ds:
        out = {
            "sst_tr":      ds["sst_tr"].values,
            "sst_va":      ds["sst_va"].values,
            "sst_te":      ds["sst_te"].values,
            "slow_tr":     ds["slow_tr"].values,
            "slow_va":     ds["slow_va"].values,
            "slow_te":     ds["slow_te"].values,
            "mu_train":    float(ds["mu_train"].values),
            "sigma_train": float(ds["sigma_train"].values),
            "split_idx":   int(ds.attrs.get("split_idx", -1)),
        }
    return out
