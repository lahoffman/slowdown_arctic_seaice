"""
08_occlusion.py — region-occlusion test for a CNN configuration (step 5.3).

For each split × seed, score the test members with the full map ('none') and
with one region zeroed (Arctic, North Pacific, tropical Pacific, North
Atlantic, everything-but-Arctic). Saves per-split NetCDF + a markdown summary
of the median drop in AUROC / AUPRC per region.

Outputs: results/occlusion[/<tag>]/occlusion_split{k}.nc, occlusion_summary.md

Usage:
  python scripts/08_occlusion.py --tag rel_aux
  python scripts/08_occlusion.py --tag rel_base --splits 0 1 --n-runs 2   # quick
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
from src.analysis import occlusion as oc
from src.cnn.splits import load_tvt_split
from src.cnn.train import load_model, model_inputs

N_SPLITS, N_RUNS = 9, 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default=None)
    p.add_argument("--splits", type=int, nargs="+", default=list(range(N_SPLITS)))
    p.add_argument("--n-runs", type=int, default=N_RUNS)
    p.add_argument("--regions", nargs="+", default=list(oc.REGIONS))
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "occlusion" / a.tag if a.tag else paths.RESULTS_DIR / "occlusion"
    out_dir.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    masks = {r: oc.region_mask(lat, lon, r) for r in a.regions}
    regions = ["none"] + a.regions
    print(f"08  —  occlusion test  tag={a.tag or 'orig'}  regions={a.regions}")
    all_ds = []
    for k in a.splits:
        sp = load_tvt_split(paths.tvt_split_path(k, a.tag))
        x_te = model_inputs(sp, "te")
        maps = x_te[0] if isinstance(x_te, list) else x_te
        y = sp["slow_te"]
        res = {"AUROC": {r: [] for r in regions}, "AUPRC": {r: [] for r in regions}}
        runs = []
        for r in range(a.n_runs):
            if not paths.model_path(k, r, a.tag).exists():
                continue
            model = load_model(paths.models_dir(a.tag), k, r); runs.append(r)
            for reg in regions:
                m = maps if reg == "none" else oc.occlude(maps, masks[reg])
                inp = [m, x_te[1]] if isinstance(x_te, list) else m
                s = oc.score(y, model.predict(inp, verbose=0).ravel())
                for met in res:
                    res[met][reg].append(s[met])
        if not runs:
            print(f"  split {k}: no models"); continue
        ds = oc.occlusion_dataset(res, regions, runs); ds.attrs.update(split_idx=k, tag=a.tag or "")
        ds.to_netcdf(out_dir / f"occlusion_split{k}.nc"); all_ds.append(ds)
        base = np.mean(res["AUROC"]["none"])
        print(f"  split {k}: AUROC full {base:.3f}  " +
              "  ".join(f"−{reg} {base - np.mean(res['AUROC'][reg]):+.3f}" for reg in a.regions), flush=True)
    if not all_ds:
        return
    stack = xr.concat(all_ds, dim="split")
    lines = ["| region occluded | ΔAUROC (median over splits, seeds) | ΔAUPRC | AUROC range |", "|---|---|---|---|"]
    full = stack["value"].sel(metric="AUROC", region="none")
    for reg in a.regions:
        d_roc = (stack["value"].sel(metric="AUROC", region=reg) - full)
        d_prc = (stack["value"].sel(metric="AUPRC", region=reg) - stack["value"].sel(metric="AUPRC", region="none"))
        v = stack["value"].sel(metric="AUROC", region=reg)
        lines.append(f"| {reg} | {float(d_roc.median()):+.3f} | {float(d_prc.median()):+.3f} | "
                     f"{float(v.min()):.2f}–{float(v.max()):.2f} |")
    md = (f"# Region occlusion — tag `{a.tag or 'orig'}`\n\nFull-map test AUROC median {float(full.median()):.3f}, "
          f"AUPRC {float(stack['value'].sel(metric='AUPRC', region='none').median()):.3f}. "
          "Negative Δ = skill lost when the region is zeroed.\n\n" + "\n".join(lines) + "\n")
    (out_dir / "occlusion_summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
