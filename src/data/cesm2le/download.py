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
        'component': 'cam.h0',
        'url_path': 'atm/proc/tseries/month_1/SST',
        'var_name': 'SST'
    },
    'AICE': {
        'component': 'cice.h',
        'url_path': 'ice/proc/tseries/month_1/aice',
        'var_name': 'aice'
    },
    'TREFHT': {
        'component': 'cam.h0',
        'url_path': 'atm/proc/tseries/month_1/TREFHT',
        'var_name': 'TREFHT'
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
