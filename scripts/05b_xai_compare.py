#!/usr/bin/env python3
"""
05b_xai_compare.py — several attribution methods on one CNN, next to occlusion (AIES §6.3).

For the chosen networks of a tag: relevance of every training sample by LRP-z,
LRP-α2β1, DeepTaylor, integrated gradients, SmoothGrad, input×gradient and (if
`shap` is installed) SHAP DeepExplainer; composites over correctly classified
slowdowns (TP); inter-method spatial correlation; each method's share of |relevance|
in the occlusion regions beside the occlusion ΔAUROC for the same regions.

Outputs: results/xai_compare/<tag>/xai_composites_split{k}_run{r}.nc, xai_summary.md,
         FIGURES_DIR/diagnostics/xai_compare_<tag>.png

Usage:
  python scripts/05b_xai_compare.py --tag rel_aux --split 2 --run 0
  python scripts/05b_xai_compare.py --tag off1_aux --split 2 5 7 --run 0 --methods lrp_z integrated_gradients smoothgrad
Runs in TF1 graph mode (iNNvestigate); separate process from training scripts.
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
from src.cnn.splits import load_tvt_split
from src.cnn.train import load_model, model_inputs
from src.xai.lrp import strip_sigmoid
from src.xai import methods as xm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True)
    p.add_argument("--split", type=int, nargs="+", default=[0]); p.add_argument("--run", type=int, nargs="+", default=[0])
    p.add_argument("--methods", nargs="+", default=xm.DEFAULT_METHODS)
    p.add_argument("--n-background", type=int, default=200, help="background samples for SHAP")
    p.add_argument("--max-samples", type=int, default=None, help="cap on training samples attributed (speed)")
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "xai_compare" / a.tag; out_dir.mkdir(parents=True, exist_ok=True)
    with nc.Dataset(paths.CESM2LE_GRID_FILE) as g:
        lat, lon = np.array(g["lat"][:]), np.array(g["lon"][:])
    with nc.Dataset(paths.LANDMASK_FILE) as d:
        ocean = np.array(d["landmask"][:]) != 1
    try:
        import shap  # noqa: F401
        have_shap = True
    except ImportError:
        have_shap = False
    methods = [m for m in a.methods if m != "shap_deep" or have_shap]
    if "shap_deep" in a.methods and not have_shap:
        print("  [skip] shap not installed — pip install shap")
    occ_dir = paths.RESULTS_DIR / "occlusion" / a.tag
    rows, comps_all = [], {}
    for k in a.split:
        sp = load_tvt_split(paths.tvt_split_path(k, a.tag))
        x_tr = model_inputs(sp, "tr"); y_tr = sp["slow_tr"].astype(int)
        multi = isinstance(x_tr, list)
        if a.max_samples:
            idx = np.random.default_rng(0).choice(y_tr.size, min(a.max_samples, y_tr.size), replace=False)
            x_tr = [v[idx] for v in x_tr] if multi else x_tr[idx]; y_tr = y_tr[idx]
        bg_idx = np.random.default_rng(1).choice(y_tr.size, min(a.n_background, y_tr.size), replace=False)
        background = [v[bg_idx] for v in x_tr] if multi else x_tr[bg_idx]
        for r in a.run:
            model = load_model(paths.models_dir(a.tag), k, r)
            prob = model.predict(x_tr, verbose=0).ravel()
            thr_file = paths.cesm2le_predictions_dir(a.tag) / f"cnn_prediction_cesm2le_M{k}_{r}.nc"
            thr = float(xr.open_dataset(thr_file)["threshold"]) if thr_file.exists() else 0.5
            tp = (y_tr == 1) & (prob >= thr); pos = y_tr == 1
            print(f"\nsplit {k} run {r}: {y_tr.size} samples, {pos.sum()} positives, {tp.sum()} TP at thr {thr:.2f}")
            logits = strip_sigmoid(model)
            comps = {}
            for m in methods:
                try:
                    rel = xm.attribute(logits, x_tr, m, background=background)
                except Exception as e:                       # one method failing must not kill the run
                    print(f"  {m:22s} FAILED: {e}"); continue
                comps[m] = dict(tp=xm.composite(rel, tp), pos=xm.composite(rel, pos),
                                share=xm.region_share(xm.composite(rel, tp, normalise=False), lat, lon, ocean=ocean))
                print(f"  {m:22s} done; Arctic share of |R| over TP {comps[m]['share']['arctic']:.2f}", flush=True)
            if not comps:
                continue
            ds = xr.Dataset({f"{m}_tp": (("lat", "lon"), comps[m]["tp"]) for m in comps}
                            | {f"{m}_pos": (("lat", "lon"), comps[m]["pos"]) for m in comps},
                            coords={"lat": lat, "lon": lon})
            ds.attrs.update(split=k, run=r, threshold=thr, n_tp=int(tp.sum()))
            ds.to_netcdf(out_dir / f"xai_composites_split{k}_run{r}.nc")
            corr = xm.spatial_correlation({m: comps[m]["tp"] for m in comps}, ocean)
            occ = None
            f = occ_dir / f"occlusion_split{k}.nc"
            if f.exists():
                o = xr.open_dataset(f)["value"].sel(metric="AUROC")
                if r in o["run"].values:
                    occ = {reg: float(o.sel(region=reg, run=r) - o.sel(region="none", run=r)) for reg in o["region"].values if reg != "none"}
            rows.append(dict(split=k, run=r, methods=list(comps), corr=corr, share={m: comps[m]["share"] for m in comps}, occ=occ))
            comps_all[(k, r)] = comps

    area = xm.region_area_share(lat, lon, ocean)
    lines = [f"# Attribution methods vs occlusion — tag `{a.tag}`\n",
             "Region share = fraction of total |relevance| (TP composite, ocean cells) inside the region; "
             "area share = fraction a uniform map would give; ΔAUROC = occlusion result for the same network.\n"]
    for row in rows:
        lines.append(f"\n## split {row['split']} run {row['run']}\n")
        lines.append("| method | " + " | ".join(f"{reg} (area {area[reg]:.2f})" for reg in area) + " |")
        lines.append("|---|" + "---|" * len(area))
        for m in row["methods"]:
            lines.append(f"| {m} | " + " | ".join(f"{row['share'][m][reg]:.2f}" for reg in area) + " |")
        if row["occ"]:
            lines.append("| **occlusion ΔAUROC** | " + " | ".join(f"{row['occ'].get(reg, float('nan')):+.3f}" for reg in area) + " |")
        lines.append("\nInter-method spatial correlation of TP composites (ocean cells):\n")
        ms = row["methods"]
        lines.append("| | " + " | ".join(ms) + " |"); lines.append("|---|" + "---|" * len(ms))
        for i, m in enumerate(ms):
            lines.append(f"| {m} | " + " | ".join(f"{row['corr'][i, j]:.2f}" for j in range(len(ms))) + " |")
    md = "\n".join(lines) + "\n"
    (out_dir / "xai_summary.md").write_text(md); print("\n" + md)
    if not a.no_fig and rows:
        from src.plotting import xai_compare as plot, style as st
        st.paper_rc()
        k, r = rows[0]["split"], rows[0]["run"]
        plot.plot_xai_compare(comps_all[(k, r)], rows[0], area, lat, lon,
                              paths.FIGURES_DIR / "diagnostics" / f"xai_compare_{a.tag}.png")


if __name__ == "__main__":
    main()
