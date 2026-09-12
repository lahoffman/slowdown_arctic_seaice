#!/usr/bin/env python3
"""
03_cesm2le_tvt_splits.py
========================
Build and save the 9 train / validate / test data splits for the JJA SST CNN.

For each of the 9 splits this script:
1. Loads JJA (June–July–August) mean SST from the monthly files produced by
   scripts/01_cesm2le_preprocessing.py, removes the forced response
   (``--demean all`` = 100-member mean, ``--demean group`` = forcing-group
   mean) and optionally lags it (``--sst-lag``).
2. Loads September slowdown labels — the original file from
   scripts/02_cesm2le_slowdowns.py or any file given with ``--labels-file``
   (e.g. the relative labels from scripts/02_cesm2le_slowdowns_relative.py).
3. Aligns SST and label years.
4. Splits the 100-member ensemble into train / validate / test blocks.
5. Standardises SST using global ocean-only statistics from the training data.
6. Applies the land mask (sets land pixels to -10).
7. Optionally adds auxiliary scalar inputs (``--aux sie_anom``: September SIE
   anomaly at the onset year, standardised with training statistics).
8. Saves each split to a NetCDF file under DATA_ROOT/results/tvt_splits[/<tag>]/.

Outputs (one per split, k = 0 … 8)
------------------------------------
    DATA_ROOT/results/tvt_splits[/<tag>]/cesm2le_sst_jja_slowdown_split{k}.nc

    Each file contains:
        sst_tr, sst_va, sst_te    — standardised, land-masked JJA SST
        slow_tr, slow_va, slow_te — binary September slowdown labels (0/1)
        mu_train, sigma_train     — normalisation statistics for downstream XAI
        aux_tr, aux_va, aux_te    — (optional) auxiliary scalar inputs
    Global attrs record labels_file, demean, sst_lag, aux and tag so that the
    downstream scripts (04/05/06) can pick the right configuration.

Usage
-----
    python scripts/03_cesm2le_tvt_splits.py                # original configuration (untagged)
    python scripts/03_cesm2le_tvt_splits.py --climate-indices-only  # indices only

    # Revision step 1.2/1.3 configurations (relative labels, group demeaning):
    LBL=$SLOWDOWN_DATA_ROOT/cesm2le/slowdowns/cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc
    python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --tag rel_base
    python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --aux sie_anom --tag rel_aux
    python scripts/03_cesm2le_tvt_splits.py --labels-file $LBL --demean group --aux sie_anom --sst-lag 1 --tag rel_lag1

Author: Lauren Hoffman  <lhoffma2@ucsc.edu>
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
from src.cnn.splits import (
    load_jja_sst_demeaned,
    iter_splits,
    block_tvt_split,
    standardize,
    standardize_aux,
    apply_landmask,
    save_tvt_split,
)
from src.analysis.baselines import load_sie_anomaly
from src.data.cesm2le.icemask import load_icemask, apply_openwater


# =============================================================================
# Configuration
# =============================================================================

# Year range — JJA of year t predicts September SIE of year t.
START_YEAR = 1990
END_YEAR   = 2040

# Slowdown variable and target month
SLOWDOWN_VAR   = 'sie'
SLOWDOWN_MONTH = 'SEP'

# Ensemble structure
N_SPLITS    = 9
N_BLOCKS    = 10
BLOCK_SIZE  = 10

# Member groups (must match keys in paths.CESM2LE_SST_MONTHLY)
MEMBER_GROUPS = ['first50', 'last50']

# NetCDF variable name for SST inside the monthly files
SST_VARNAME = 'sst_mon'

# Climate index files (output of scripts/02_cesm2le_climate_indices.py)
CLIMATE_INDICES_DIR = paths.CESM2LE_DIR / 'climate_indices'


# =============================================================================
# Helpers
# =============================================================================

def load_landmask() -> np.ndarray:
    """Load the CESM2-LE land mask.  0 = ocean, 1 = land."""
    with nc.Dataset(paths.LANDMASK_FILE, 'r') as ds:
        return np.array(ds.variables['landmask'][:])


def load_slowdown_labels(
    fpath: Path,
    start_year: int,
    end_year: int,
) -> np.ndarray:
    """
    Load binary September slowdown labels for the requested year range.

    ``fpath`` is the original file from scripts/02_cesm2le_slowdowns.py or a
    relative-label file from scripts/02_cesm2le_slowdowns_relative.py; both
    store ``slowdown(nens, nyr)`` where ``nyr`` is the *starting year* of each
    trend window, so selecting years [start_year, end_year] gives the label for
    the September SIE trend beginning in each of those years.

    Returns
    -------
    slowdown : np.ndarray
        Binary array, shape ``(nens, nyear)``.  1 = slowdown, 0 = normal.
    """
    fpath = Path(fpath)
    if not fpath.exists():
        raise FileNotFoundError(
            f"Slowdown file not found:\n  {fpath}\n"
            f"Run scripts/02_cesm2le_slowdowns.py (or _relative.py) first."
        )

    with xr.open_dataset(fpath) as ds:
        slowdown = (
            ds['slowdown']
            .sel(nyr=slice(start_year, end_year))
            .values                                    # (nens, nyear)
        )

    print(f"  Slowdown labels: shape {slowdown.shape}, "
          f"years {start_year}–{end_year}, "
          f"prevalence {slowdown.mean():.3f}")
    return slowdown.astype(np.int8)


def load_climate_indices_jja(
    start_year: int,
    end_year: int,
) -> dict:
    """
    Load JJA-mean CESM2-LE climate indices for the TVT year range.

    Returns a dict with keys ``'nino34'``, ``'ipo'``, ``'arctic'``, each
    an array of shape ``(nens, nyear)`` containing the JJA seasonal mean
    of the corresponding climate index.  Any index whose source file is
    missing is silently skipped.

    Parameters
    ----------
    start_year, end_year : int
        Year range to select (must match the SST / slowdown range).
    """
    index_specs = [
        ('nino34', 'cesm2le_nino34_index.nc',     'nino34_months'),
        ('ipo',    'cesm2le_ipo_index.nc',         'ipo_filtered'),
        ('arctic', 'cesm2le_arctic_sst_index.nc',  'arctic_sst_months'),
    ]
    out = {}
    for key, fname, varname in index_specs:
        fpath = CLIMATE_INDICES_DIR / fname
        if not fpath.exists():
            print(f'  [skip] {fname} not found — {key} index will be omitted')
            continue
        with xr.open_dataset(fpath) as ds:
            arr = ds[varname].sel(nyr=slice(start_year, end_year)).values
        out[key] = np.nanmean(arr[:, :, 5:8], axis=2)   # JJA mean → (nens, nyear)
        print(f'  {key:8s} JJA: shape {out[key].shape}')
    return out


def save_climate_indices_split(
    indices_split: dict,
    split_idx: int,
    savepath,
) -> None:
    """
    Save split-aligned JJA-mean climate index arrays to NetCDF.

    Parameters
    ----------
    indices_split : dict
        Keys like ``'nino34_tr'``, ``'ipo_va'``, ``'arctic_te'``, each a 1-D
        float array with one value per sample.
    split_idx : int
        TVT split index (0–8).
    savepath : Path or str
        Output NetCDF path.
    """
    from pathlib import Path
    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)

    data_vars = {}
    for idx_name in ('nino34', 'ipo', 'arctic'):
        for part in ('tr', 'va', 'te'):
            key = f'{idx_name}_{part}'
            if key in indices_split:
                dim = f'n{part}'
                data_vars[key] = ((dim,), indices_split[key].astype(np.float32))

    if not data_vars:
        print(f'  [skip] No climate indices to save for split {split_idx}')
        return

    ds = xr.Dataset(data_vars)
    ds.attrs['split_idx'] = split_idx
    ds.attrs['description'] = (
        f'JJA-mean climate indices aligned with TVT split {split_idx}. '
        'Indices: nino34 (Niño3.4), ipo (IPO filtered), arctic (Arctic SST).'
    )
    for v in ds.data_vars:
        ds[v].attrs['long_name'] = (
            v.replace('nino34', 'Niño3.4 JJA')
             .replace('ipo', 'IPO filtered JJA')
             .replace('arctic', 'Arctic SST JJA')
             .replace('_tr', ' (train)')
             .replace('_va', ' (validation)')
             .replace('_te', ' (test)')
        )

    encoding = {v: {'zlib': True, 'complevel': 4} for v in ds.data_vars}
    ds.to_netcdf(savepath, format='NETCDF4', encoding=encoding)
    print(f'    Climate indices saved → {savepath}')


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build TVT data splits for the JJA SST CNN.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--climate-indices-only', '-c',
        action='store_true',
        default=False,
        help='Only split and save climate indices (skip SST / slowdown / '
             'landmask processing).  Block assignments are deterministic, so '
             'no existing split files are needed.',
    )
    parser.add_argument('--labels-file', type=Path, default=None,
                        help='Slowdown label NetCDF (default: original '
                             '02_cesm2le_slowdowns output for --variable/--month).')
    parser.add_argument('--demean', choices=['all', 'group'], default='all',
                        help="Forced response removed from SST (and from the SIE "
                             "auxiliary input): 100-member mean (default) or forcing-group mean.")
    parser.add_argument('--sst-lag', type=int, default=0,
                        help='Use JJA SST of year t − lag for target year t (default 0).')
    parser.add_argument('--aux', choices=['none', 'sie_anom'], default='none',
                        help="Auxiliary scalar input stored alongside the maps "
                             "(default none; 'sie_anom' = September SIE anomaly at onset).")
    parser.add_argument('--openwater', action='store_true',
                        help='Zero the SST anomaly where JJA ice concentration > 0.15 '
                             '(needs 02_cesm2le_icemask.py; revision step 1.6).')
    parser.add_argument('--ice-threshold', type=float, default=0.15)
    parser.add_argument('--no-fig', action='store_true', help='Skip the open-water check figure.')
    parser.add_argument('--tag', default=None,
                        help='Configuration tag → outputs go to tvt_splits/<tag>/ '
                             '(default: untagged original location).')
    parser.add_argument('--start-year', type=int, default=START_YEAR)
    parser.add_argument('--end-year', type=int, default=END_YEAR,
                        help='Last onset year (default 2040; use 2030 to cap onsets).')
    parser.add_argument('--variable', default=SLOWDOWN_VAR, choices=['sie', 'sia'])
    parser.add_argument('--month', default=SLOWDOWN_MONTH)
    return parser.parse_args()


# =============================================================================
# Main — climate-indices-only (lightweight fast path)
# =============================================================================

def main_climate_indices_only(args: argparse.Namespace) -> None:
    """
    Replay the block assignments from _get_block_indices and split climate
    indices *without* loading SST, slowdown, or the land mask.
    """
    from src.cnn.splits import _get_block_indices
    start_year, end_year, tag = args.start_year, args.end_year, args.tag

    print()
    print('=' * 70)
    print('03  —  CESM2-LE TVT Splits  (climate indices only)')
    print('=' * 70)
    print(f'  Data root    : {paths.DATA_ROOT}')
    print(f'  Years        : {start_year}–{end_year}')
    print(f'  N splits     : {N_SPLITS}')
    print(f'  Output dir   : {paths.tvt_splits_dir(tag)}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # 1.  Load climate indices
    # ------------------------------------------------------------------
    print('\n[1] Loading CESM2-LE climate indices (JJA mean) ...')
    climate_indices = load_climate_indices_jja(start_year, end_year)
    if not climate_indices:
        print('No climate indices found — nothing to do.')
        return

    # ------------------------------------------------------------------
    # 2.  Replay block assignments and split
    # ------------------------------------------------------------------
    print(f'\n[2] Splitting indices across {N_SPLITS} TVT splits ...\n')

    for k, test_block, val_block, train_blocks in _get_block_indices(N_SPLITS, N_BLOCKS):
        print(f'  Split {k}  '
              f'(test={test_block}, val={val_block}, '
              f'train={train_blocks})')

        idx_split = {}
        for idx_name, idx_arr in climate_indices.items():
            tr, va, te = block_tvt_split(
                idx_arr, train_blocks, val_block, test_block,
                N_BLOCKS, BLOCK_SIZE,
            )
            idx_split[f'{idx_name}_tr'] = tr
            idx_split[f'{idx_name}_va'] = va
            idx_split[f'{idx_name}_te'] = te

        save_climate_indices_split(
            idx_split, k, paths.climate_indices_split_path(k, tag),
        )

    print()
    print('=' * 70)
    print('Done.  Climate index splits saved to:')
    print(f'  {paths.tvt_splits_dir(tag)}')
    print('=' * 70 + '\n')


# =============================================================================
# Main — full pipeline
# =============================================================================

def main(args: argparse.Namespace) -> None:
    start_year, end_year, tag = args.start_year, args.end_year, args.tag
    labels_file = args.labels_file or paths.cesm2le_slowdown_file(args.variable, args.month)
    out_dir = paths.tvt_splits_dir(tag)

    print()
    print('=' * 70)
    print('03  —  CESM2-LE TVT Splits')
    print('=' * 70)
    print(f'  Data root    : {paths.DATA_ROOT}')
    print(f'  Target years : {start_year}–{end_year}')
    print(f'  Labels       : {labels_file}')
    print(f'  Demean       : {args.demean}')
    print(f'  SST lag      : {args.sst_lag} yr')
    print(f'  Aux input    : {args.aux}')
    print(f'  Open water   : {"aice > %g masked" % args.ice_threshold if args.openwater else "no"}')
    print(f'  Tag          : {tag or "(none — original configuration)"}')
    print(f'  N splits     : {N_SPLITS}')
    print(f'  Output dir   : {out_dir}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # 1.  Load JJA SST  (forced response removed, optionally lagged)
    # ------------------------------------------------------------------
    print('\n[1] Loading JJA SST ...')
    sst, sst_years = load_jja_sst_demeaned(
        sst_monthly_template=paths.CESM2LE_SST_MONTHLY,
        member_groups=MEMBER_GROUPS,
        start_year=start_year,
        end_year=end_year,
        sst_varname=SST_VARNAME,
        demean=args.demean,
        sst_lag=args.sst_lag,
    )
    print(f'    SST shape : {sst.shape}  '
          f'(nens={sst.shape[0]}, nyear={sst.shape[1]}, '
          f'nx={sst.shape[2]}, ny={sst.shape[3]})')

    # ------------------------------------------------------------------
    # 1b.  Open-water variant: zero the anomaly under JJA sea ice (step 1.6)
    # ------------------------------------------------------------------
    if args.openwater:
        if not paths.CESM2LE_ICEMASK_JJA.exists():
            raise FileNotFoundError(f"{paths.CESM2LE_ICEMASK_JJA} not found — run "
                                    "scripts/02_cesm2le_icemask.py first.")
        ice = load_icemask(paths.CESM2LE_ICEMASK_JJA, sst_years - args.sst_lag, args.ice_threshold)
        with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
            arctic_rows = np.array(g['lat'][:]) >= 65
        sst_before = sst if not args.no_fig else None
        sst = apply_openwater(sst, ice)
        print(f'    open-water mask applied: {ice.mean():.3f} of all cells, '
              f'{ice[:, :, arctic_rows, :].mean():.3f} of cells north of 65°N set to zero anomaly')
        if not args.no_fig:
            from src.plotting import icemask as plot_im, style as st
            st.paper_rc()
            with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
                glat, glon = np.array(g['lat'][:]), np.array(g['lon'][:])
            out_png = paths.FIGURES_DIR / 'diagnostics' / f'openwater_check_{tag or "untagged"}.png'
            plot_im.plot_openwater_check(sst_before, sst, ice, glat, glon, sst_years - args.sst_lag,
                                         out_png, member=6, year=2010, landmask=load_landmask())
            del sst_before

    # ------------------------------------------------------------------
    # 2.  Load September slowdown labels
    # ------------------------------------------------------------------
    print(f'\n[2] Loading slowdown labels ({labels_file.name}) ...')
    slowdown = load_slowdown_labels(labels_file, start_year, end_year)

    # Sanity check — ensemble and year dimensions must match
    if sst.shape[0] != slowdown.shape[0]:
        raise ValueError(
            f"Ensemble size mismatch: SST has {sst.shape[0]} members, "
            f"slowdown has {slowdown.shape[0]}."
        )
    if sst.shape[1] != slowdown.shape[1]:
        raise ValueError(
            f"Year mismatch: SST has {sst.shape[1]} years, "
            f"slowdown has {slowdown.shape[1]}."
        )

    # ------------------------------------------------------------------
    # 2b.  Load climate indices (JJA mean, optional)
    # ------------------------------------------------------------------
    print(f'\n[2b] Loading CESM2-LE climate indices (JJA mean) ...')
    climate_indices = load_climate_indices_jja(start_year, end_year)
    for key, arr in climate_indices.items():
        if arr.shape[0] != sst.shape[0] or arr.shape[1] != sst.shape[1]:
            raise ValueError(
                f"Climate index '{key}' shape {arr.shape} does not match "
                f"SST shape ({sst.shape[0]}, {sst.shape[1]})."
            )

    # ------------------------------------------------------------------
    # 2c.  Auxiliary scalar input (onset-year SIE anomaly), optional
    # ------------------------------------------------------------------
    aux_fields, aux_names = None, []
    if args.aux == 'sie_anom':
        print(f'\n[2c] Loading auxiliary input: September SIE anomaly (demean={args.demean}) ...')
        _, sie_anom = load_sie_anomaly(paths.CESM2LE_AICE_DIR / 'metrics', sst_years,
                                       month=args.month, variable=args.variable,
                                       demean=args.demean)
        aux_fields, aux_names = [sie_anom], ['sie_anom']
        print(f'    sie_anom : {sie_anom.shape}  std {np.nanstd(sie_anom):.3f} M km²')

    # ------------------------------------------------------------------
    # 3.  Load land mask
    # ------------------------------------------------------------------
    print('\n[3] Loading land mask ...')
    landmask = load_landmask()
    print(f'    Landmask shape : {landmask.shape}  '
          f'(ocean fraction: {(landmask == 0).mean():.3f})')

    # ------------------------------------------------------------------
    # 4.  Loop over splits
    # ------------------------------------------------------------------
    print(f'\n[4] Building {N_SPLITS} TVT splits ...\n')

    for split in iter_splits(sst, slowdown, N_SPLITS, N_BLOCKS, BLOCK_SIZE):
        k = split['split_idx']
        blocks = (split['train_blocks'], split['val_block'], split['test_block'])
        print(f'  Split {k}  '
              f'(test block={split["test_block"]}, '
              f'val block={split["val_block"]}, '
              f'train blocks={split["train_blocks"]})')

        # Standardise using training ocean-only statistics
        sst_tr_std, sst_va_std, sst_te_std, mu, sigma = standardize(
            split['sst_tr'], split['sst_va'], split['sst_te'], landmask
        )
        print(f'    μ_train = {mu:.4f},  σ_train = {sigma:.4f}   '
              f'prevalence tr/va/te = {split["slow_tr"].mean():.3f}/'
              f'{split["slow_va"].mean():.3f}/{split["slow_te"].mean():.3f}')

        # Apply land mask (land → -10 in standardised space)
        sst_tr_m = apply_landmask(sst_tr_std, landmask)
        sst_va_m = apply_landmask(sst_va_std, landmask)
        sst_te_m = apply_landmask(sst_te_std, landmask)

        # Auxiliary scalars: same block split, standardised with training stats
        aux, aux_attrs = None, {}
        if aux_fields:
            cols = [block_tvt_split(f, *blocks, N_BLOCKS, BLOCK_SIZE) for f in aux_fields]
            a_tr = np.stack([c[0] for c in cols], axis=1)
            a_va = np.stack([c[1] for c in cols], axis=1)
            a_te = np.stack([c[2] for c in cols], axis=1)
            a_tr, a_va, a_te, a_mu, a_sd = standardize_aux(a_tr, a_va, a_te)
            aux = {'tr': a_tr, 'va': a_va, 'te': a_te}
            aux_attrs = {'aux_mu_train': ','.join(f'{v:.6g}' for v in a_mu),
                         'aux_sigma_train': ','.join(f'{v:.6g}' for v in a_sd)}

        # Save
        save_tvt_split(
            sst_tr=sst_tr_m,
            sst_va=sst_va_m,
            sst_te=sst_te_m,
            slow_tr=split['slow_tr'],
            slow_va=split['slow_va'],
            slow_te=split['slow_te'],
            mu_train=mu,
            sigma_train=sigma,
            split_idx=k,
            savepath=paths.tvt_split_path(k, tag),
            attrs={
                'sst_years':       f'{start_year - args.sst_lag}-{end_year - args.sst_lag}',
                'target_years':    f'{start_year}-{end_year}',
                'slowdown_var':    args.variable,
                'slowdown_month':  args.month,
                'labels_file':     str(labels_file),
                'demean':          args.demean,
                'sst_lag':         int(args.sst_lag),
                'aux':             args.aux,
                'openwater':       int(args.openwater),
                'ice_threshold':   float(args.ice_threshold) if args.openwater else 0.0,
                'tag':             tag or '',
                'member_groups':   str(MEMBER_GROUPS),
                **aux_attrs,
            },
            aux=aux,
            aux_names=aux_names or None,
        )

        # Split and save climate indices (same block assignment)
        if climate_indices:
            idx_split = {}
            for idx_name, idx_arr in climate_indices.items():
                tr, va, te = block_tvt_split(idx_arr, *blocks, N_BLOCKS, BLOCK_SIZE)
                idx_split[f'{idx_name}_tr'] = tr
                idx_split[f'{idx_name}_va'] = va
                idx_split[f'{idx_name}_te'] = te
            save_climate_indices_split(
                idx_split, k, paths.climate_indices_split_path(k, tag),
            )

    print()
    print('=' * 70)
    print('Done.  Splits saved to:')
    print(f'  {out_dir}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    args = parse_args()
    if args.climate_indices_only:
        main_climate_indices_only(args)
    else:
        main(args)
