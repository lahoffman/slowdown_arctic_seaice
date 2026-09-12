#!/usr/bin/env python3
"""
02_cesm2le_forced.py
====================
Compute and save CESM2-LE forced-response fields.

Currently computes:
  - Ensemble-mean JJA SST on the full 2D atmospheric grid (192 x 288).
    This is the forced component subtracted from each ensemble member
    during CNN training.  Persisting it here lets the observation pipeline
    (``scripts/03_ersst_test.py``) reuse the same field without reloading
    all 100 members.
  - Forcing-group-mean JJA SST (revision step 1.3): the same field per
    forcing group (cmip6 = members 0-49, smbb = 50-99).  Used when the TVT
    splits are built with ``--demean group`` and for the observation
    pipeline once a group is chosen as the forced reference.
  - Diagnostic figures (FIGURES_DIR/diagnostics/):
      forced_group_difference.png   SMBB - CMIP6 forced JJA SST map + Arctic series
      forced_demeaned_arctic.png    member Arctic SST anomalies under both demeanings

Requires
--------
Monthly SST NetCDF files from ``scripts/01_cesm2le_preprocessing.py``:
    DATA_ROOT/cesm2le/sst/mon/sst_cesmle_{group}members_mon_{MON}_199001-210012.nc

Output
------
    DATA_ROOT/cesm2le/forced/cesm2le_ensmean_jja_sst.nc
    DATA_ROOT/cesm2le/forced/cesm2le_groupmean_jja_sst.nc

Usage
-----
    python scripts/02_cesm2le_forced.py                  # ensmean + groupmean + figures
    python scripts/02_cesm2le_forced.py --demean all     # original ensmean only
    python scripts/02_cesm2le_forced.py --demean group --no-fig

Author: Lauren Hoffman
Email:  lhoffma2@ucsc.edu
"""

import argparse
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.data.cesm2le.forced import (
    load_jja_sst_all_members,
    forced_response,
    group_means,
    save_ensmean_jja_sst,
    save_groupmean_jja_sst,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute CESM2-LE forced-response fields.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--start-year', type=int, default=1990,
        help='First year in the SST data (default: 1990).',
    )
    parser.add_argument(
        '--end-year', type=int, default=2100,
        help='Last year in the SST data (default: 2100).',
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Override output file path (default: paths.CESM2LE_ENSMEAN_JJA).',
    )
    parser.add_argument(
        '--demean', choices=['all', 'group', 'both'], default='both',
        help="Which forced field(s) to write: 'all' = 100-member mean, "
             "'group' = per forcing group, 'both' (default).",
    )
    parser.add_argument('--no-fig', action='store_true', help='Skip diagnostic figures.')
    parser.add_argument('--period', type=int, nargs=2, default=[2000, 2020],
                        help='Averaging period for the difference map (default 2000 2020).')
    return parser.parse_args()


def _load_lat_lon():
    import netCDF4 as nc
    with nc.Dataset(paths.CESM2LE_GRID_FILE, 'r') as ds:
        return np.array(ds.variables['lat'][:]), np.array(ds.variables['lon'][:])


def _load_landmask():
    import netCDF4 as nc
    if not paths.LANDMASK_FILE.exists():
        return None
    with nc.Dataset(paths.LANDMASK_FILE, 'r') as ds:
        return np.array(ds.variables['landmask'][:])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    years = np.arange(args.start_year, args.end_year + 1)
    output = Path(args.output) if args.output else paths.CESM2LE_ENSMEAN_JJA
    output_group = paths.CESM2LE_GROUPMEAN_JJA
    sst_monthly_dir = paths.CESM2LE_SST_DIR / 'mon'

    print('\n' + '=' * 70)
    print('CESM2-LE Forced Response')
    print('=' * 70)
    print(f'  SST dir  : {sst_monthly_dir}')
    print(f'  Years    : {args.start_year}–{args.end_year}')
    print(f'  Demean   : {args.demean}')
    print(f'  Output   : {output}' + (f'\n             {output_group}' if args.demean != 'all' else ''))
    print('=' * 70)

    print('\nLoading JJA SST for all members ...')
    sst_jja = load_jja_sst_all_members(str(sst_monthly_dir))   # (nens, nyear, nlat, nlon)
    if sst_jja.shape[1] != years.size:
        raise ValueError(f'SST has {sst_jja.shape[1]} years but --start/--end-year give {years.size}.')

    ensmean = np.nanmean(sst_jja, axis=0)
    groupmean, names = group_means(sst_jja)

    print('\nSaving ...')
    if args.demean in ('all', 'both'):
        save_ensmean_jja_sst(ensmean, years, str(output))
    if args.demean in ('group', 'both'):
        save_groupmean_jja_sst(groupmean, names, years, str(output_group))

    if not args.no_fig:
        from src.plotting import forced as plot
        from src.plotting import style as st
        st.paper_rc()
        lat, lon = _load_lat_lon()
        landmask = _load_landmask()
        out_dir = paths.FIGURES_DIR / 'diagnostics'
        print('\nDiagnostic figures ...')
        plot.plot_group_forced_difference(groupmean, names, years, ensmean, lat, lon,
                                          out_dir / 'forced_group_difference.png',
                                          period=tuple(args.period), landmask=landmask)
        dem_all = sst_jja - forced_response(sst_jja, 'all')
        dem_grp = sst_jja - forced_response(sst_jja, 'group')
        if landmask is not None:
            dem_all = np.where(landmask == 1, np.nan, dem_all)
            dem_grp = np.where(landmask == 1, np.nan, dem_grp)
        plot.plot_demeaned_arctic_index(dem_all, dem_grp, years, lat,
                                        out_dir / 'forced_demeaned_arctic.png')

    print('\n' + '=' * 70)
    print('Done!')
    print(f'  ensmean   : {ensmean.shape}  (nyear, nlat, nlon)')
    print(f'  groupmean : {groupmean.shape}  (ngroup, nyear, nlat, nlon)  {names}')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
