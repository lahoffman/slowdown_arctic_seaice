"""
make_figure.py — build manuscript figures from cached outputs.

Loading lives here (via configs.paths); drawing lives in src/plotting/paper.py;
statistics in src/analysis. Figures are saved to FIGURES_DIR/paper/.

Usage:
  python scripts/make_figure.py --list
  python scripts/make_figure.py S1                          # relative labels (the default) → fig_S1.pdf
  python scripts/make_figure.py S1 --labels original --suffix _orig
  python scripts/make_figure.py 2 S10 S11 S12 S13 regional   # composites share one pass over the data
  python scripts/make_figure.py 4 --forced-method ensmean
  python scripts/make_figure.py all --fmt pdf

Figure ids: 1 2 3 4 S1 … S15, S17 S18 S19 plus extras phase_all, regional, sie_gmt_joint, learning_curve, learning_curves, occlusion.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import composites as cmp, phase_stats as ps
from src.cnn.splits import load_tvt_split
from src.data.cesm2le.slowdowns import load_sie_monthly_files
from src.plotting import paper, style as st

MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
N_SPLITS, N_SEEDS, BLOCK_SIZE, START_YEAR = 9, 5, 10, 1990
METRIC_KEYS = ["Accuracy", "Precision", "Recall", "F1"]


# =============================================================================
# Shared loaders (cached per run so figures can share one pass)
# =============================================================================

class Data:
    """Lazy, memoised access to everything the figures need."""

    def __init__(self, a):
        self.a = a
        self._cache = {}

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    # -- labels / SIE ----------------------------------------------------------
    def label_file(self):
        a = self.a
        if a.labels == "original":
            return paths.cesm2le_slowdown_file(a.variable, a.month)
        mode = "" if a.sigma_mode == "pooled" else f"_{a.sigma_mode}"
        off = f"_off{a.trend_offset}" if a.trend_offset else ""
        return paths.CESM2LE_SLOWDOWNS_DIR / (
            f"cesm2le_{a.variable}_slowdown_relative_{a.month}_w{a.window}_s{a.n_sigma:g}_{a.demean}{mode}{off}_1990-2100.nc")

    # -- CNN configuration (--tag): splits / predictions / metrics / attributions --
    @property
    def tag(self):
        return self.a.tag

    def pred_dir(self):
        return paths.cesm2le_predictions_dir(self.tag)

    def labels(self):
        return self.get("labels", lambda: xr.open_dataset(self.label_file()).load())

    def labels_original(self):
        """Original (LB22-style) labels for contrast in S2; None if the file is missing."""
        f = paths.cesm2le_slowdown_file(self.a.variable, self.a.month)
        return self.get("labels_orig", lambda: xr.open_dataset(f).load() if f.exists() else None)

    # -- revision additions: sweep (S17), product comparison (S18), forced removal (S19) --
    def grid(self):
        def _load():
            import netCDF4 as nc
            with nc.Dataset(paths.CESM2LE_GRID_FILE) as ds:
                lat, lon = np.array(ds["lat"][:]), np.array(ds["lon"][:])
            landmask = None
            if paths.LANDMASK_FILE.exists():
                with nc.Dataset(paths.LANDMASK_FILE) as ds:
                    landmask = np.array(ds["landmask"][:])
            return lat, lon, landmask
        return self.get("grid", _load)

    def sweep(self):
        return self.get("sweep", lambda: xr.open_dataset(paths.RESULTS_DIR / "sensitivity" / "sweep.nc").load())

    def products(self):
        def _load():
            from src.analysis import obs_products as op
            lat, lon, landmask = self.grid()
            e = op.load_product(paths.ERSST_REGRIDDED, "ERSSTv5"); o = op.load_product(paths.OISST_REGRIDDED, "OISST v2.1")
            return op.compare_jja(e, o, lat, 1990, self.a.obs_end_year, landmask=landmask)
        return self.get("products", _load)

    def forced_removal(self):
        def _load():
            from src.data.observations import obs_input as oi
            from src.data.observations.ersst.climate_indices import _correct_forced_mean_and_trend
            from src.plotting.forced import arctic_mean
            lat, lon, landmask = self.grid()
            years = np.arange(1990, self.a.obs_end_year + 1)
            mask = lambda f: np.where(landmask == 1, np.nan, f) if landmask is not None else f
            forced = {m: arctic_mean(mask(oi.forced_sst_field(m, years, paths.CESM2LE_ENSMEAN_JJA, paths.CESM2LE_GROUPMEAN_JJA)), lat)
                      for m in ("ensmean", "group_cmip6", "group_smbb")}
            methods = ["linear", "quadratic", "ensmean", "group_cmip6", "group_smbb"]
            prods, series = {}, {}
            for name, path in paths.OBS_PRODUCTS.items():
                if not path.exists():
                    continue
                jja = mask(oi.jja_by_year(oi.load_monthly_product(path), years))
                ok = ~np.isnan(jja).all(axis=(1, 2)); yrs = years[ok]; obs = arctic_mean(jja[ok], lat)
                t = yrs - yrs.mean(); prods[name] = (yrs, obs); series[name] = {}
                for m in methods:
                    if m in ("linear", "quadratic"):
                        series[name][m] = obs - np.polyval(np.polyfit(t, obs, 1 if m == "linear" else 2), t)
                    else:
                        series[name][m] = _correct_forced_mean_and_trend(obs, forced[m][ok], np.arange(yrs.size, dtype=float))[2]
            return dict(prods=prods, series=series, forced_arctic=forced, years=years, methods=methods, claim=(2016, 2025))
        return self.get("forced_removal", _load)

    # -- forced response per forcing group (S3; 02_cesm2le_forced.py) -----------
    def forced(self):
        def _load():
            import netCDF4 as nc
            from src.data.cesm2le.forced import load_groupmean_jja_sst
            gm, names, years = load_groupmean_jja_sst(paths.CESM2LE_GROUPMEAN_JJA)
            with nc.Dataset(paths.CESM2LE_GRID_FILE) as ds:
                lat, lon = np.array(ds["lat"][:]), np.array(ds["lon"][:])
            landmask = None
            if paths.LANDMASK_FILE.exists():
                with nc.Dataset(paths.LANDMASK_FILE) as ds:
                    landmask = np.array(ds["landmask"][:])
            return dict(groupmean=gm, names=names, years=years, lat=lat, lon=lon, landmask=landmask,
                        period=tuple(self.a.forced_period))
        return self.get("forced", _load)

    def sie(self):
        return self.get("sie", lambda: load_sie_monthly_files(
            str(paths.CESM2LE_AICE_DIR / "metrics"), self.a.month, variable=self.a.variable,
            start_year=1990, end_year=2100))

    def nsidc(self):
        def _load():
            a = self.a; m = MONTHS.index(a.month)
            thr_f = paths.NSIDC_SIE_SLOWDOWN_THRESHOLDS if a.variable == "sie" else paths.NSIDC_SIA_SLOWDOWN_THRESHOLDS
            ev_f = (paths.nsidc_sie_slowdown_events if a.variable == "sie" else paths.nsidc_sia_slowdown_events)(m + 1)
            with xr.open_dataset(thr_f) as ds:
                thr, frac = float(ds["threshold_slowdown"].values[m]), float(ds["fraction_slowdown"].values[m])
            with xr.open_dataset(ev_f) as ds:
                out = dict(trend_years=ds["year"].values.astype(int), trends=ds["linear_trend"].values,
                           slowdown=ds["slowdown"].values.astype(int), ice=ds["seaice"].values,
                           ice_years=ds.coords["yearice"].values.astype(int))
            return {**out, "threshold": thr, "mean_trend": thr / frac}
        return self.get("nsidc", _load)

    # -- one split × seed (S4, S5, S7, learning curve) ---------------------------
    def single(self):
        def _load():
            k, r = self.a.split, self.a.seed
            sp = load_tvt_split(paths.tvt_split_path(k, self.tag))
            with xr.open_dataset(self.pred_dir() / f"cnn_prediction_cesm2le_M{k}_{r}.nc") as ds:
                score = {p: ds[f"y_prob_{p}"].values for p in ("train", "val", "test")}
                thr = float(ds["threshold"].values)
            hist_p = (paths.LOGS_DIR / self.tag if self.tag else paths.LOGS_DIR) / f"history_split{k}_run{r}.json"
            hist = json.load(open(hist_p)) if hist_p.exists() else None
            return dict(y_true={"train": sp["slow_tr"], "val": sp["slow_va"], "test": sp["slow_te"]},
                        y_score=score, threshold=thr, history=hist)
        return self.get("single", _load)

    def occlusion(self, tags=None):
        """Stacked occlusion results for every configuration that has them (``--occlusion-tags``)."""
        tags = tags or self.a.occlusion_tags
        def _load():
            out = {}
            for t in tags:
                files = sorted((paths.RESULTS_DIR / "occlusion" / t).glob("occlusion_split*.nc"))
                if files:
                    out[t] = xr.concat([xr.open_dataset(f) for f in files], dim="split")
            if not out:
                raise FileNotFoundError("no results/occlusion/<tag>/occlusion_split*.nc")
            return out
        return self.get("occlusion", _load)

    def histories(self):
        """Training histories of every saved model for this tag."""
        d = paths.LOGS_DIR / self.tag if self.tag else paths.LOGS_DIR
        files = sorted(d.glob("history_split*_run*.json"))
        if not files:
            raise FileNotFoundError(f"no history_split*_run*.json in {d}")
        return self.get("histories", lambda: [json.load(open(f)) for f in files])

    # -- metrics for all splits (S6) --------------------------------------------
    def metrics(self):
        def _load():
            vals = {k: [] for k in METRIC_KEYS}
            for k in range(N_SPLITS):
                with xr.open_dataset(paths.metrics_path(k, self.tag)) as ds:
                    for m in METRIC_KEYS:
                        vals[m].append(ds["metric_value"].sel(metric=m, split="test").values)
            return {m: np.concatenate(v) for m, v in vals.items()}
        return self.get("metrics", _load)

    # -- composites + phase stats: one streaming pass over splits × seeds ----------
    def pass_over_splits(self):
        def _load():
            acc = cmp.CompositeAccumulator(positive_only=False)
            acc_pos = cmp.CompositeAccumulator(scenarios=("TP",), positive_only=True)
            phase = {p: {k: [] for k in ps.PHASE_SPECS} for p in ("train", "test")}
            slow = {p: [] for p in ("train", "test")}
            tp = {p: [[] for _ in range(N_SEEDS)] for p in ("train", "test")}
            for k in range(N_SPLITS):
                sp = load_tvt_split(paths.tvt_split_path(k, self.tag))
                y = {"train": sp["slow_tr"], "test": sp["slow_te"]}
                with xr.open_dataset(paths.climate_indices_split_path(k, self.tag)) as ds:
                    for p, suf in (("train", "tr"), ("test", "te")):
                        for key in ps.PHASE_SPECS:
                            phase[p][key].append(ds[f"{key}_{suf}"].values)
                for p in ("train", "test"):
                    slow[p].append(y[p] == 1)
                for r in range(N_SEEDS):
                    pf = self.pred_dir() / f"cnn_prediction_cesm2le_M{k}_{r}.nc"
                    if not pf.exists():
                        print(f"  [skip] {pf.name}"); continue
                    with xr.open_dataset(pf) as ds:
                        sc = {"train": ds["y_prob_train"].values, "test": ds["y_prob_test"].values}
                    for p in ("train", "test"):
                        thr = cmp.pr_threshold(y[p], sc[p])
                        tp[p][r].append(((sc[p] >= thr) & (y[p] == 1)))
                    lf = paths.attribution_path(k, r, self.tag)
                    if lf.exists():
                        lrp, lat, lon = cmp.load_lrp(lf)
                        acc.add(sp["sst_tr"], y["train"], sc["train"], lrp, lat, lon)
                        acc_pos.add(sp["sst_tr"], y["train"], sc["train"], lrp, lat, lon)
                    else:
                        acc.add(sp["sst_tr"], y["train"], sc["train"])
                print(f"  split {k} done")
            out = dict(acc=acc, acc_pos=acc_pos, summaries={})
            for p in ("train", "test"):
                tp_seeds = [np.concatenate(seed_masks) for seed_masks in tp[p] if seed_masks]
                out["summaries"][p] = [
                    ps.phase_summary(np.concatenate(phase[p][key]), np.concatenate(slow[p]),
                                     tp_seeds, key, self.a.n_boot)
                    for key in ("arctic", "ipo", "nino34")]
            return out
        return self.get("pass", _load)

    # -- observations (Fig. 4) --------------------------------------------------
    def observations(self):
        def _load():
            a = self.a
            with xr.open_dataset(paths.ersst_testing_file(a.forced_method)) as ds:
                obs_years = ds["years"].values.astype(int)
            probs = []
            pdir = paths.ersst_predictions_dir(a.forced_method)
            for k in range(N_SPLITS):
                for r in range(N_SEEDS):
                    f = pdir / f"cnn_prediction_ersst_M{k}_{r}.nc"
                    if f.exists():
                        with xr.open_dataset(f) as ds:
                            probs.append(ds["y_prob"].values)
            probs = np.array(probs)
            frac = (probs >= 0.5).mean(0)
            with xr.open_dataset(paths.nsidc_sie_slowdown_events(9)) as ds:
                sy, se = ds["year"].values.astype(int), ds["slowdown"].values.astype(int)

            def jja(vals, start_year):
                n = vals.size // 12
                return np.nanmean(vals[:n * 12].reshape(n, 12)[:, 5:8], 1), np.arange(start_year, start_year + n)

            import pandas as pd
            with xr.open_dataset(paths.ERSST_ARCTIC_SST) as ds:
                t = pd.DatetimeIndex(ds["time"].values); v = ds["arctic_sst"].values
                sel = (t.year >= obs_years[0]) & (t.year <= obs_years[-1])
                arctic, ay = jja(v[sel], obs_years[0])
            with xr.open_dataset(paths.ERSST_NINO34) as ds:
                v = ds["nino34"].values
            t = pd.date_range("1854-01-01", periods=v.size, freq="MS")
            sel = (t.year >= obs_years[0]) & (t.year <= obs_years[-1])
            nino, ny = jja(v[sel], obs_years[0])
            with xr.open_dataset(paths.ERSST_IPO) as ds:
                v = ds["ipo_filtered"].values
            t = pd.date_range("1854-01-01", periods=v.size, freq="MS")
            sel = (t.year >= obs_years[0]) & (t.year <= obs_years[-1])
            ipo, iy = jja(v[sel], obs_years[0])
            indices = {"arctic": (ay, arctic, "Arctic SST index", (1, -1)),
                       "nino34": (ny, nino, "Niño 3.4 index", (0.4, -0.4)),
                       "ipo": (iy, ipo, "IPO index", ())}
            single = probs[0] if a.single_model else None
            return dict(obs_years=obs_years, fraction=frac, obs_slow_years=sy, obs_slow=se,
                        indices=indices, single_prob=single,
                        title=f"forced removal: {a.forced_method}")
        return self.get("obs", _load)

    # -- baselines (S3) -----------------------------------------------------------
    def baselines(self):
        def _load():
            a = self.a
            tag = a.baselines_tag or a.tag or ("" if a.labels == "original" else f"rel_w{a.window}_s{a.n_sigma:g}")
            f = paths.RESULTS_DIR / "baselines" / tag / "baselines_all_splits.nc"
            if not f.exists():
                raise FileNotFoundError(f"{f} — run 07_baselines.py" + (f" --tag {tag}" if tag else ""))
            return xr.open_dataset(f).load()
        return self.get("baselines", _load)

    # -- SIE vs GMT (S13) -------------------------------------------------------
    def sie_gmt(self):
        def _load():
            with xr.open_dataset(paths.cesm2le_slowdown_file("sie", self.a.month)) as ds:
                s, sy, s_tr = ds["slowdown"].values, ds["nyr"].values, ds["linear_trends_ens"].values
            with xr.open_dataset(paths.cesm2le_gmt_slowdown_file()) as ds:
                g, gy, g_tr = ds["slowdown"].values, ds["nyr"].values, ds["gmt_trends_ens"].values
            yrs = np.intersect1d(sy, gy)
            si, gi = np.isin(sy, yrs), np.isin(gy, yrs)
            s, g = s[:, si], g[:, gi]
            return dict(years=yrs, sie_count=s.sum(0), gmt_count=g.sum(0),
                        both_count=((s == 1) & (g == 1)).sum(0),
                        gmt_tr=g_tr[:, gi].ravel(), sie_tr=s_tr[:, si].ravel())
        return self.get("sie_gmt", _load)


# =============================================================================
# Figure builders: id → (function producing the Figure, description)
# =============================================================================

def _ts_args(d: Data):
    sie, years = d.sie()
    return dict(nsidc=d.nsidc(), sie=sie, years=years, labels=d.labels(), member=d.a.member,
                window=d.a.window if d.a.labels == "relative" else 10,
                varname=d.a.variable.upper(), month=d.a.month)


def build_1(d):
    img = None
    if d.a.schematic and Path(d.a.schematic).exists():
        from PIL import Image
        img = np.asarray(Image.open(d.a.schematic))
    return paper.fig_1(schematic=img, **_ts_args(d))


def build_composite(scenario, **kw):
    def _b(d):
        comp = d.pass_over_splits()["acc"].result(scenario)
        if "lrp" not in comp:
            raise FileNotFoundError("No LRP attributions found — run 05_cesm2le_lrp.py")
        return paper.fig_2(comp, **kw)
    return _b


def build_s9(d):
    acc = d.pass_over_splits()["acc"]
    return paper.fig_s10({k: acc.result(k) for k in ("ALL_SLOW", "TP", "ALL_NONSLOW", "TN")})


def build_regional(d):
    return paper.fig_regional_relevance(d.pass_over_splits()["acc_pos"].result("TP"))


def build_s4(d):
    sie, years = d.sie()
    return paper.fig_s4(sie, years, d.labels(), varlabel=f"{d.a.month} {d.a.variable.upper()}")


def build_s8(d):
    s = d.single()
    yt, ys = s["y_true"]["test"], s["y_score"]["test"]
    years = np.arange(START_YEAR, START_YEAR + yt.size // BLOCK_SIZE)
    members = np.arange(d.a.split * BLOCK_SIZE, (d.a.split + 1) * BLOCK_SIZE)
    return paper.fig_s9(yt, (ys >= s["threshold"]).astype(int), years, members)


FIGURES = {
    "1":   (build_1, "schematic + NSIDC record + one member"),
    "2":   (build_composite("TP"), "TP composite: SST + LRP (signed)"),
    "3":   (lambda d: paper.fig_3(d.pass_over_splits()["summaries"]["test"]), "P(TP | phase), test"),
    "4":   (lambda d: paper.fig_4(**d.observations()), "observations: CNN votes + indices"),
    "S1":  (lambda d: paper.fig_s1(**_ts_args(d)), "slowdown definition, 6 panels"),
    "S2":  (lambda d: paper.fig_s2(d.labels(), d.labels_original(), d.a.cap_year), "pooled σ and onset cap"),
    "S3":  (lambda d: paper.fig_s3(**d.forced()), "forced response: SMBB − CMIP6 (02_cesm2le_forced.py)"),
    "S4":  (build_s4, "label distributions, 8 panels"),
    "S5":  (lambda d: paper.fig_s5(d.baselines()), "baselines vs CNN skill (07_baselines.py)"),
    "S6":  (lambda d: paper.fig_s6(d.single()["y_true"]["test"], d.single()["y_score"]["test"]), "PR curve"),
    "S7":  (lambda d: paper.fig_s7(d.single()["y_true"], d.single()["y_score"], d.single()["threshold"]), "confusion matrices"),
    "S8":  (lambda d: paper.fig_s8(d.metrics()), "metric strip, all CNNs"),
    "S9":  (build_s8, "test-member slowdown timeline"),
    "S10":  (build_s9, "SST composites: all vs CNN-filtered"),
    "S11":  (build_composite("FP"), "FP composite"),
    "S12": (build_composite("TN"), "TN composite"),
    "S13": (build_composite("FN"), "FN composite"),
    "S14": (lambda d: paper.fig_s14(d.pass_over_splits()["summaries"]["train"]), "P(event | phase), train, all vs TP"),
    "S17": (lambda d: paper.fig_s17(d.sweep(), ["logit_sie_anom", "logit_indices", "logit_sie_pacific", "logit_all_scalars"]), "label sensitivity heat-maps (09_sensitivity_sweep.py)"),
    "S18": (lambda d: paper.fig_s18(d.products(), *d.grid()), "ERSST vs OISST on the CESM2 grid"),
    "S19": (lambda d: paper.fig_s19(**d.forced_removal()), "observed Arctic index vs forced reference"),
    "S15": (lambda d: paper.fig_s15(*(d.sie_gmt()[k] for k in ("years", "sie_count", "gmt_count", "both_count"))), "SIE vs GMT slowdown counts"),
    "phase_all":      (lambda d: paper.fig_phase_all(d.pass_over_splits()["summaries"]["train"]), "P(slowdown | phase), all slowdowns"),
    "regional":       (build_regional, "TP composite with region boxes + regional relevance"),
    "sie_gmt_joint":  (lambda d: paper.fig_sie_gmt_joint(d.sie_gmt()["gmt_tr"], d.sie_gmt()["sie_tr"]), "joint PDF of GMT and SIE trends"),
    "learning_curve": (lambda d: paper.fig_learning_curve(d.single()["history"]), "loss vs epoch, one CNN"),
    "learning_curves": (lambda d: paper.fig_learning_curves(d.histories()), "loss vs epoch, all 45 CNNs of a tag"),
    "occlusion": (lambda d: paper.fig_occlusion(d.occlusion()), "region occlusion, all configurations"),
}


def parse_args(argv=None):
    """Parse CLI options; pass a list (e.g. ``[]`` or ``["--labels", "relative"]``) from notebooks."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("figures", nargs="*", help="figure ids (see --list) or 'all'")
    p.add_argument("--list", action="store_true")
    p.add_argument("--labels", default="relative", choices=["original", "relative"],
                   help="slowdown label definition (default: relative, the manuscript's)")
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--trend-offset", type=int, default=0,
                   help="relative labels whose window starts k years after onset (file suffix _off<k>); 0 = onset-inclusive")
    p.add_argument("--occlusion-tags", nargs="+",
                   default=["rel_base", "rel_aux", "rel_openwater", "rel_lag1", "rel_concurrent"],
                   help="configurations shown in the occlusion figure")
    p.add_argument("--n-sigma", type=float, default=1.0)
    p.add_argument("--demean", default="group", choices=["group", "all"])
    p.add_argument("--sigma-mode", default="pooled", choices=["pooled", "yearly"],
                   help="σ mode of the relative label file (step 1.2)")
    p.add_argument("--tag", default=None,
                   help="CNN configuration tag (tvt_splits/models/predictions/attributions/<tag>); "
                        "default: original CNN outputs")
    p.add_argument("--member", type=int, default=6, help="highlighted member (Figs 1, S1)")
    p.add_argument("--cap-year", type=int, default=2030, help="onset cap marked in S2")
    p.add_argument("--obs-end-year", type=int, default=2025, help="last year of observations (S18, S19)")
    p.add_argument("--forced-period", type=int, nargs=2, default=[2000, 2020],
                   help="averaging period for the S3 forced-difference map")
    p.add_argument("--split", type=int, default=0, help="split for single-model figures (S4, S5, S7)")
    p.add_argument("--seed", type=int, default=0, help="seed index for single-model figures")
    p.add_argument("--variable", default="sie", choices=["sie", "sia"])
    p.add_argument("--month", default="SEP")
    p.add_argument("--forced-method", default="ensmean", choices=["ensmean", "linear"])
    p.add_argument("--single-model", action="store_true", help="Fig 4: add the single-CNN panel")
    p.add_argument("--schematic", default=str(paths.DATA_ROOT / "models/schematic/cnn_schematic.png"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--baselines-tag", default=None,
                   help="results/baselines/<tag> to plot in S3 (default: derived from --labels)")
    p.add_argument("--suffix", default="", help="appended to output names, e.g. _orig")
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args(argv)


def main():
    a = parse_args()
    if a.list or not a.figures:
        for k, (_, desc) in FIGURES.items():
            print(f"  {k:15s} {desc}")
        return
    ids = list(FIGURES) if a.figures == ["all"] else a.figures
    st.paper_rc()
    d = Data(a)
    out_dir = a.out_dir or paths.FIGURES_DIR / "paper"
    tag = a.suffix
    for fid in ids:
        build, _ = FIGURES[fid]
        print(f"[{fid}]")
        try:
            fig = build(d)
        except FileNotFoundError as e:
            print(f"  skipped: {e}"); continue
        st.save(fig, out_dir / f"fig_{fid}{tag}.{a.fmt}", dpi=a.dpi)


if __name__ == "__main__":
    main()
