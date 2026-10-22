#!/usr/bin/env python3
"""
11_slide_onset_example.py — data for the onset-coupling schematic (one member, one onset).

Picks a member-year with a large September SIE anomaly at onset whose onset-inclusive
trend (t … t+9) is much steeper than the offset trend (t+1 … t+10), and saves its SIE
anomaly series and JJA SST anomaly map (both relative to the forcing-group mean).

Output: RESULTS_DIR/slides/onset_example.npz

Usage:
  python scripts/11_slide_onset_example.py [--member M --onset YYYY]
"""

import argparse
import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl
from src.analysis.interannual import trend_anomaly_offset
from src.cnn.splits import load_jja_sst_demeaned


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member", type=int, default=None)
    p.add_argument("--onset", type=int, default=None)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--zmin", type=float, default=2.0, help="minimum onset SIE anomaly (in σ) for auto-pick")
    return p.parse_args()


def main():
    a = parse_args()
    years = np.arange(1990, 2061)
    onsets = np.arange(1995, 2031)
    sie, sie_anom = bl.load_sie_anomaly(paths.CESM2LE_AICE_DIR / "metrics", years, demean="group")

    tr0 = trend_anomaly_offset(sie, years, onsets, a.window, 0)
    tr1 = trend_anomaly_offset(sie, years, onsets, a.window, 1)
    i_on = np.searchsorted(years, onsets)
    x0 = sie_anom[:, i_on]
    z = x0 / np.nanstd(x0)

    if a.member is None or a.onset is None:
        score = np.where(z >= a.zmin, tr1 - tr0, -np.inf)       # inclusive steep, offset flat
        m, k = np.unravel_index(np.nanargmax(score), score.shape)
    else:
        m, k = a.member, int(np.searchsorted(onsets, a.onset))
    t = int(onsets[k])
    print(f"member {m}, onset {t}: SIE anom {x0[m, k]:.3f} ({z[m, k]:.1f}σ), "
          f"trend anom t…t+{a.window - 1} {tr0[m, k]:.4f}, t+1…t+{a.window} {tr1[m, k]:.4f} (per yr)")

    sst, _ = load_jja_sst_demeaned(paths.CESM2LE_SST_MONTHLY, start_year=t, end_year=t, demean="group")
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        land = np.array(d["landmask"][:]) == 1

    out_dir = paths.RESULTS_DIR / "slides"; out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "onset_example.npz"
    np.savez_compressed(out, member=m, onset=t, window=a.window, years=years,
                        sie=sie[m], sie_anom=sie_anom[m], sie_anom_sd=np.nanstd(x0),
                        trend0=tr0[m, k], trend1=tr1[m, k],
                        sst=np.asarray(sst[m, 0], dtype=np.float32), lat=lat, lon=lon, land=land)
    print(f"  → {out}")


if __name__ == "__main__":
    main()
