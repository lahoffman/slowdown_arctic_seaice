#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2-LE Data Combining and Organization

Functions to combine raw CESM2-LE data chunks and separate by month.

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
from pathlib import Path
from typing import List, Optional

from .download import CMIP6_MEMBERS, SMBB_MEMBERS


def combine_ensemble_members(
    variable: str,
    raw_data_path: str,
    output_path: str,
    start_years: Optional[List[str]] = None,
    end_years: Optional[List[str]] = None,
    hist_scenario: str = 'cmip6',
    future_scenario: str = 'cmip6',
    hist_cutoff_idx: int = 3,
    member_group: str = 'first50',
    component: str = 'cam',
    frequency: str = 'h0'
) -> None:
    """
    Combine raw CESM2-LE data chunks into a single file with shape [nens, ntime, nlat, nlon].

    The CESM2-LE data comes in chunks by ensemble member and time period. This function
    loads all chunks for a given set of ensemble members and concatenates them along
    the time dimension.

    Parameters
    ----------
    variable : str
        Variable name (e.g., 'SST', 'AICE', 'TS')
    raw_data_path : str
        Base directory where raw CESM2 data is stored
    output_path : str
        Full path for the output NetCDF file
    start_years : List[str], optional
        List of start years for each time chunk.
        Default: ['1990','2000','2010','2015','2025','2035','2045','2055','2065','2075','2085','2095']
    end_years : List[str], optional
        List of end years for each time chunk.
        Default: ['1999','2009','2014','2024','2034','2044','2054','2064','2074','2084','2094','2100']
    hist_scenario : str, optional
        Historical scenario name (default: 'cmip6')
    future_scenario : str, optional
        Future scenario name (default: 'cmip6' for SSP370)
    hist_cutoff_idx : int, optional
        Index where historical period ends and future begins (default: 3)
    member_group : str, optional
        Which members to process: 'first50', 'last50', or 'all' (default: 'first50')
    component : str, optional
        Model component (default: 'cam')
    frequency : str, optional
        Output frequency (default: 'h0')

    Returns
    -------
    None
        Saves combined data to output_path

    Examples
    --------
    >>> combine_ensemble_members(
    ...     variable='SST',
    ...     raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
    ...     output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_first50_199001-210012.nc',
    ...     member_group='first50'
    ... )
    """
    # Default time periods (1990-2100)
    if start_years is None:
        start_years = ['1990','2000','2010','2015','2025','2035','2045','2055','2065','2075','2085','2095']
    if end_years is None:
        end_years = ['1999','2009','2014','2024','2034','2044','2054','2064','2074','2084','2094','2100']

    # Select member list based on group
    if member_group == 'first50':
        members = CMIP6_MEMBERS
    elif member_group == 'last50':
        members = SMBB_MEMBERS
    elif member_group == 'all':
        members = CMIP6_MEMBERS + SMBB_MEMBERS
    else:
        raise ValueError(f"member_group must be 'first50', 'last50', or 'all', got {member_group}")

    # Variable name in file (lowercase)
    var_lower = variable.lower()
    var_upper = variable.upper()

    # Process each ensemble member
    data_list = []
    n_members = len(members)

    for i, member in enumerate(members):
        print(f"Processing member {i+1}/{n_members}: {member}")

        member_data = []
        # Loop through time chunks
        for j, (start_year, end_year) in enumerate(zip(start_years, end_years)):

            # Determine scenario and file pattern
            if j < hist_cutoff_idx:
                # Historical period
                if member_group == 'first50':
                    scenario = f'BHIST{hist_scenario}'
                    filename = f'b.e21.{scenario}.f09_g17.LE2-{member}.{component}.{frequency}.{var_upper}.{start_year}01-{end_year}12.nc'
                else:  # last50 uses smbb
                    scenario = f'BHISTsmbb'
                    filename = f'b.e21.{scenario}.f09_g17.LE2-{member}.{component}.{frequency}.{var_upper}.{start_year}01-{end_year}12.nc'
            else:
                # Future period (SSP370)
                if member_group == 'first50':
                    scenario = f'BSSP370{future_scenario}'
                    filename = f'b.e21.{scenario}.f09_g17.LE2-{member}.{component}.{frequency}.{var_upper}.{start_year}01-{end_year}12.nc'
                else:  # last50 uses smbb
                    scenario = f'BSSP370smbb'
                    filename = f'b.e21.{scenario}.f09_g17.LE2-{member}.{component}.{frequency}.{var_upper}.{start_year}01-{end_year}12.nc'

            filepath = Path(raw_data_path) / filename
            # Some variables use lowercase in the filename (e.g. 'aice')
            # while others use uppercase (e.g. 'SST').  Fall back to lowercase.
            if not filepath.exists():
                filename = filename.replace(f'.{var_upper}.', f'.{var_lower}.')
                filepath = Path(raw_data_path) / filename

            # Load data chunk
            try:
                dataset = nc.Dataset(filepath, 'r')
                # Variable names differ by component: CAM uses uppercase (e.g.
                # 'SST'), CICE uses lowercase (e.g. 'aice').  Try both.
                if var_lower in dataset.variables:
                    data_key = var_lower
                elif var_upper in dataset.variables:
                    data_key = var_upper
                else:
                    available = [v for v in dataset.variables
                                 if v not in ('time', 'lat', 'lon', 'lev',
                                              'time_bnds', 'date', 'datesec')]
                    dataset.close()
                    raise KeyError(
                        f"Variable '{variable}' not found in {filepath.name}. "
                        f"Available data variables: {available}"
                    )
                # Use raw[:] to get a masked array so that fill values
                # (e.g. 1e30 on land points in CICE output) are properly
                # masked.  np.array() alone ignores the mask, leaving huge
                # fill values that cause overflow in downstream calculations.
                raw = dataset.variables[data_key][:]
                if hasattr(raw, 'filled'):
                    chunk_data = raw.filled(np.nan).astype(np.float32)
                else:
                    chunk_data = np.array(raw, dtype=np.float32)
                dataset.close()
                member_data.append(chunk_data)
            except FileNotFoundError:
                print(f"Warning: File not found: {filepath}")
                raise

        # Concatenate all time chunks for this member
        member_timeseries = np.concatenate(member_data, axis=0)
        data_list.append(member_timeseries)

    # Stack all ensemble members
    combined_data = np.array(data_list)

    # Create year array
    start_year_int = int(start_years[0])
    end_year_int = int(end_years[-1])
    unique_years = np.arange(start_year_int, end_year_int + 1)

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            var_lower: (("nem", "nm", "nx", "ny"), combined_data),
            "unique_years": (("nyr",), unique_years),
        },
        coords={
            "nem": np.arange(combined_data.shape[0]),  # Ensemble members
            "nm": np.arange(combined_data.shape[1]),   # Time (monthly)
            "nx": np.arange(combined_data.shape[2]),   # Latitude
            "ny": np.arange(combined_data.shape[3]),   # Longitude
            "nyr": np.arange(unique_years.shape[0]),   # Years
        },
    )

    # Add attributes
    ds.attrs['description'] = f'CESM2-LE {variable} data combined from raw chunks'
    ds.attrs['member_group'] = member_group
    ds.attrs['n_members'] = combined_data.shape[0]
    ds.attrs['time_range'] = f'{start_years[0]}-{end_years[-1]}'

    # Save to NetCDF with compression
    encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
    ds.to_netcdf(output_path, format="NETCDF4", encoding=encoding)

    print(f"\nNetCDF file saved to {output_path}")
    print(f"Shape: {combined_data.shape} [nens, ntime, nlat, nlon]")


def separate_by_month(
    combined_file: str,
    output_dir: str,
    variable: str,
    start_year: int = 1990,
    end_year: int = 2100,
    member_label: str = 'first50members'
) -> None:
    """
    Separate combined monthly data into individual files for each month.

    Takes a file with shape [nens, ntime, nlat, nlon] where ntime includes all months
    from start_year to end_year, and creates 12 separate files (one per month) with
    shape [nens, nyears, nlat, nlon].

    Parameters
    ----------
    combined_file : str
        Path to combined NetCDF file (output from combine_ensemble_members)
    output_dir : str
        Directory where monthly files will be saved
    variable : str
        Variable name (e.g., 'SST', 'AICE')
    start_year : int, optional
        First year in the dataset (default: 1990)
    end_year : int, optional
        Last year in the dataset (default: 2100)
    member_label : str, optional
        Label for output filename (default: 'first50members')

    Returns
    -------
    None
        Saves 12 monthly NetCDF files to output_dir

    Examples
    --------
    >>> separate_by_month(
    ...     combined_file='/cofast/lhoffman/cesmle/sst/mon_combined/sst_first50_199001-210012.nc',
    ...     output_dir='/cofast/lhoffman/cesmle/sst/mon',
    ...     variable='SST',
    ...     member_label='first50members'
    ... )
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load combined data
    print(f"Loading combined file: {combined_file}")
    dataset = nc.Dataset(combined_file, 'r')
    var_lower = variable.lower()

    # Get variable name in file (might be var_lower or f'{var_lower}_mon' or just var_lower)
    if var_lower in dataset.variables:
        data_var = var_lower
    elif f'{var_lower}_mon' in dataset.variables:
        data_var = f'{var_lower}_mon'
    else:
        # Try to find it
        var_names = list(dataset.variables.keys())
        data_vars = [v for v in var_names if v not in ['nem', 'nm', 'nx', 'ny', 'nyr', 'unique_years']]
        if len(data_vars) == 1:
            data_var = data_vars[0]
        else:
            raise ValueError(f"Could not determine data variable. Available: {var_names}")

    data = np.array(dataset.variables[data_var])
    dataset.close()

    # Create time array
    time = np.arange(
        np.datetime64(f'{start_year}-01'),
        np.datetime64(f'{end_year+1}-01'),
        np.timedelta64(1, 'M')
    )
    months = np.array([t.astype(object).month for t in time])

    # Month labels
    month_labels = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                    'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

    # Process each month
    for month_num in range(1, 13):
        print(f"Processing {month_labels[month_num-1]}...")

        # Extract this month's data
        idx = np.where(months == month_num)[0]
        monthly_data = data[:, idx, :, :]

        # Create output filename
        output_file = Path(output_dir) / f'{var_lower}_cesmle_{member_label}_mon_{month_labels[month_num-1]}_{start_year}01-{end_year}12.nc'

        # Create xarray Dataset
        ds = xr.Dataset(
            {
                f"{var_lower}_mon": (("nem", "nm", "nx", "ny"), monthly_data),
            },
            coords={
                "nem": np.arange(monthly_data.shape[0]),  # Ensemble members
                "nm": np.arange(monthly_data.shape[1]),   # Years
                "nx": np.arange(monthly_data.shape[2]),   # Latitude
                "ny": np.arange(monthly_data.shape[3]),   # Longitude
            },
        )

        # Add attributes
        ds.attrs['description'] = f'CESM2-LE {variable} data for {month_labels[month_num-1]}'
        ds.attrs['month'] = month_labels[month_num-1]
        ds.attrs['month_number'] = month_num
        ds.attrs['time_range'] = f'{start_year}-{end_year}'

        # Save to NetCDF with compression
        encoding = {var: {"zlib": True, "complevel": 4} for var in ds.data_vars}
        ds.to_netcdf(output_file, format="NETCDF4", encoding=encoding)

        print(f"  Saved: {output_file}")
        print(f"  Shape: {monthly_data.shape} [nens, nyears, nlat, nlon]")

    print(f"\nAll monthly files saved to {output_dir}")


def process_cesmle_variable(
    variable: str,
    raw_data_path: str,
    combined_output_path: str,
    monthly_output_dir: str,
    member_groups: List[str] = ['first50', 'last50'],
    **kwargs
) -> None:
    """
    Complete pipeline: combine raw data and separate by month for a given variable.

    This is a convenience function that runs both combine_ensemble_members and
    separate_by_month for one or more ensemble member groups.

    Parameters
    ----------
    variable : str
        Variable name (e.g., 'SST', 'AICE')
    raw_data_path : str
        Base directory where raw CESM2 data is stored
    combined_output_path : str
        Path template for combined output files (use {group} placeholder)
    monthly_output_dir : str
        Directory for monthly output files
    member_groups : List[str], optional
        Which groups to process (default: ['first50', 'last50'])
    **kwargs
        Additional arguments passed to combine_ensemble_members and separate_by_month

    Examples
    --------
    >>> process_cesmle_variable(
    ...     variable='SST',
    ...     raw_data_path='/cofast/lhoffman/cesmle/sst/raw',
    ...     combined_output_path='/cofast/lhoffman/cesmle/sst/mon_combined/sst_{group}_199001-210012.nc',
    ...     monthly_output_dir='/cofast/lhoffman/cesmle/sst/mon'
    ... )
    """
    for group in member_groups:
        print(f"\n{'='*60}")
        print(f"Processing {group} ensemble members for {variable}")
        print(f"{'='*60}\n")

        # Combine data
        combined_file = combined_output_path.format(group=group)
        combine_ensemble_members(
            variable=variable,
            raw_data_path=raw_data_path,
            output_path=combined_file,
            member_group=group,
            **{k: v for k, v in kwargs.items() if k in ['start_years', 'end_years', 'hist_scenario', 'future_scenario']}
        )

        # Separate by month
        member_label = f'{group}members'
        separate_by_month(
            combined_file=combined_file,
            output_dir=monthly_output_dir,
            variable=variable,
            member_label=member_label,
            **{k: v for k, v in kwargs.items() if k in ['start_year', 'end_year']}
        )

    print(f"\n{'='*60}")
    print(f"Complete! Processed {variable} for all member groups")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Example usage
    print("CESM2-LE Data Processing Module")
    print("Import this module and use the functions:")
    print("  - combine_ensemble_members(): Combine raw data chunks")
    print("  - separate_by_month(): Separate combined data by month")
    print("  - process_cesmle_variable(): Complete pipeline")
