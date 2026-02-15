#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CESM2 Large Ensemble Data Processing Module

This module provides functions to download and process CESM2-LE data:
1. Combine raw data chunks (by ensemble member and year range) into full timeseries
2. Separate combined data by month for easier analysis

The CESM2-LE dataset comes in chunks organized by:
- Ensemble member (100 total: first 50 are cmip6, last 50 are smbb)
- Time periods (typically spanning 1990-2100 with different year ranges)
- Variables (SST, AICE, etc.)

Author: Lauren Hoffman
Email: lhoffma2@ucsc.edu
"""

import numpy as np
import netCDF4 as nc
import xarray as xr
import subprocess
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict


# CESM2-LE Ensemble member definitions
CMIP6_MEMBERS = [
    1001.001, 1021.002, 1041.003, 1061.004, 1081.005, 1101.006, 1121.007, 1141.008, 1161.009, '1181.010',
    1231.001, 1231.002, 1231.003, 1231.004, 1231.005, 1231.006, 1231.007, 1231.008, 1231.009, '1231.010',
    1251.001, 1251.002, 1251.003, 1251.004, 1251.005, 1251.006, 1251.007, 1251.008, 1251.009, '1251.010',
    1281.001, 1281.002, 1281.003, 1281.004, 1281.005, 1281.006, 1281.007, 1281.008, 1281.009, '1281.010',
    1301.001, 1301.002, 1301.003, 1301.004, 1301.005, 1301.006, 1301.007, 1301.008, 1301.009, '1301.010'
]

SMBB_MEMBERS = [
    1231.011, 1231.012, 1231.013, 1231.014, 1231.015, 1231.016, 1231.017, 1231.018, 1231.019, '1231.020',
    1251.011, 1251.012, 1251.013, 1251.014, 1251.015, 1251.016, 1251.017, 1251.018, 1251.019, '1251.020',
    1281.011, 1281.012, 1281.013, 1281.014, 1281.015, 1281.016, 1281.017, 1281.018, 1281.019, '1281.020',
    1301.011, 1301.012, 1301.013, 1301.014, 1301.015, 1301.016, 1301.017, 1301.018, 1301.019, '1301.020',
    1011.001, 1031.002, 1051.003, 1071.004, 1091.005, 1111.006, 1131.007, 1151.008, 1171.009, '1191.010'
]

# Variable configuration: maps variable names to their model component and URL paths
VARIABLE_CONFIG = {
    'SST': {
        'component': 'pop.h',
        'url_path': 'ocn/proc/tseries/month_1/SST',
        'var_name': 'SST'
    },
    'AICE': {
        'component': 'cice.h',
        'url_path': 'ice/proc/tseries/month_1/aice',
        'var_name': 'aice'
    },
    'TS': {
        'component': 'cam.h0',
        'url_path': 'atm/proc/tseries/month_1/TS',
        'var_name': 'TS'
    },
    'PRECT': {
        'component': 'cam.h0',
        'url_path': 'atm/proc/tseries/month_1/PRECT',
        'var_name': 'PRECT'
    },
}

# Base URL for CESM2-LE data
BASE_URL = "https://osdf-director.osg-htc.org/ncar/gdex/d651056/CESM2-LE"


def download_raw_data(
    variable: str,
    output_dir: str,
    member_groups: List[str] = ['first50', 'last50'],
    hist_startyears: Optional[List[int]] = None,
    hist_endyears: Optional[List[int]] = None,
    ssp_startyears: Optional[List[int]] = None,
    ssp_endyears: Optional[List[int]] = None,
    component: Optional[str] = None,
    url_path: Optional[str] = None,
    var_name: Optional[str] = None,
    dry_run: bool = False
) -> None:
    """
    Download raw CESM2-LE data from UCAR using wget.

    This function downloads all time chunks for specified ensemble member groups
    directly from the UCAR CESM2-LE repository.

    Parameters
    ----------
    variable : str
        Variable name (e.g., 'SST', 'AICE', 'TS')
    output_dir : str
        Directory where raw files will be saved
    member_groups : List[str], optional
        Which member groups to download: 'first50', 'last50', or both (default: both)
    hist_startyears : List[int], optional
        Start years for historical period (default: [1990, 2000, 2010])
    hist_endyears : List[int], optional
        End years for historical period (default: [1999, 2009, 2014])
    ssp_startyears : List[int], optional
        Start years for SSP370 period (default: [2015, 2025, ..., 2095])
    ssp_endyears : List[int], optional
        End years for SSP370 period (default: [2024, 2034, ..., 2100])
    component : str, optional
        Model component (e.g., 'pop.h', 'cice.h'). If None, looked up from VARIABLE_CONFIG
    url_path : str, optional
        URL path component. If None, looked up from VARIABLE_CONFIG
    var_name : str, optional
        Variable name in filename. If None, looked up from VARIABLE_CONFIG
    dry_run : bool, optional
        If True, print commands without executing (default: False)

    Returns
    -------
    None
        Downloads files to output_dir

    Examples
    --------
    >>> download_raw_data(
    ...     variable='SST',
    ...     output_dir='/cofast/lhoffman/cesmle/sst/raw',
    ...     member_groups=['first50']
    ... )
    """
    # Default time periods
    if hist_startyears is None:
        hist_startyears = [1990, 2000, 2010]
    if hist_endyears is None:
        hist_endyears = [1999, 2009, 2014]
    if ssp_startyears is None:
        ssp_startyears = [2015, 2025, 2035, 2045, 2055, 2065, 2075, 2085, 2095]
    if ssp_endyears is None:
        ssp_endyears = [2024, 2034, 2044, 2054, 2064, 2074, 2084, 2094, 2100]

    # Get variable configuration
    if variable.upper() in VARIABLE_CONFIG:
        config = VARIABLE_CONFIG[variable.upper()]
        component = component or config['component']
        url_path = url_path or config['url_path']
        var_name = var_name or config['var_name']
    else:
        # User must provide these manually for custom variables
        if not all([component, url_path, var_name]):
            raise ValueError(
                f"Variable '{variable}' not in VARIABLE_CONFIG. "
                f"Must provide component, url_path, and var_name explicitly."
            )

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each member group
    for group in member_groups:
        if group == 'first50':
            members = CMIP6_MEMBERS
            label = 'cmip6'
        elif group == 'last50':
            members = SMBB_MEMBERS
            label = 'smbb'
        else:
            raise ValueError(f"member_group must be 'first50' or 'last50', got {group}")

        print(f"\n{'='*70}")
        print(f"Downloading {variable} data for {group} members ({label})")
        print(f"{'='*70}\n")

        # Download for each member
        for member_idx, member in enumerate(members):
            print(f"\nMember {member_idx+1}/{len(members)}: {member}")

            # Historical period
            for startyear, endyear in zip(hist_startyears, hist_endyears):
                _download_single_file(
                    member=member,
                    label=label,
                    variable=var_name,
                    component=component,
                    url_path=url_path,
                    startyear=startyear,
                    endyear=endyear,
                    scenario='BHIST',
                    output_dir=output_dir,
                    dry_run=dry_run
                )

            # SSP370 period
            for startyear, endyear in zip(ssp_startyears, ssp_endyears):
                _download_single_file(
                    member=member,
                    label=label,
                    variable=var_name,
                    component=component,
                    url_path=url_path,
                    startyear=startyear,
                    endyear=endyear,
                    scenario='BSSP370',
                    output_dir=output_dir,
                    dry_run=dry_run
                )

    print(f"\n{'='*70}")
    print(f"Download complete! Files saved to {output_dir}")
    print(f"{'='*70}\n")


def _download_single_file(
    member: str,
    label: str,
    variable: str,
    component: str,
    url_path: str,
    startyear: int,
    endyear: int,
    scenario: str,
    output_dir: str,
    dry_run: bool = False
) -> None:
    """
    Download a single CESM2-LE file using wget.

    Parameters
    ----------
    member : str
        Ensemble member ID
    label : str
        Member label ('cmip6' or 'smbb')
    variable : str
        Variable name in filename
    component : str
        Model component (e.g., 'pop.h', 'cice.h')
    url_path : str
        URL path component
    startyear : int
        Start year
    endyear : int
        End year
    scenario : str
        Scenario name ('BHIST' or 'BSSP370')
    output_dir : str
        Output directory
    dry_run : bool
        If True, print command without executing
    """
    # Construct filename
    filename = f"b.e21.{scenario}{label}.f09_g17.LE2-{member}.{component}.{variable}.{startyear}01-{endyear}12.nc"

    # Check if file already exists
    output_path = Path(output_dir) / filename
    if output_path.exists():
        print(f"  ✓ Already exists: {filename}")
        return

    # Construct download URL
    url = f"{BASE_URL}/{url_path}/{filename}"

    # Build wget command
    wget_cmd = ['wget', '-P', output_dir, url]

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(wget_cmd)}")
        return

    # Execute download
    print(f"  Downloading: {filename}")
    try:
        result = subprocess.run(
            wget_cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Sometimes wget adds .1 suffix if file exists, rename it
        downloaded_with_suffix = Path(output_dir) / f"{filename}.1"
        if downloaded_with_suffix.exists():
            downloaded_with_suffix.rename(output_path)

        print(f"  ✓ Downloaded: {filename}")

    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error downloading {filename}")
        print(f"    {e.stderr}")
        raise


def add_variable_config(
    variable: str,
    component: str,
    url_path: str,
    var_name: Optional[str] = None
) -> None:
    """
    Add a new variable configuration to VARIABLE_CONFIG.

    Use this to add support for additional CESM2 variables not in the default config.

    Parameters
    ----------
    variable : str
        Variable name (uppercase, e.g., 'UBOT')
    component : str
        Model component (e.g., 'cam.h0', 'pop.h')
    url_path : str
        URL path component (e.g., 'atm/proc/tseries/month_1/UBOT')
    var_name : str, optional
        Variable name in filename (if different from variable)

    Examples
    --------
    >>> add_variable_config(
    ...     variable='UBOT',
    ...     component='cam.h0',
    ...     url_path='atm/proc/tseries/month_1/UBOT'
    ... )
    """
    VARIABLE_CONFIG[variable.upper()] = {
        'component': component,
        'url_path': url_path,
        'var_name': var_name or variable
    }
    print(f"Added configuration for {variable}")


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

            # Load data chunk
            try:
                dataset = nc.Dataset(filepath, 'r')
                chunk_data = np.array(dataset.variables[var_lower])
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
