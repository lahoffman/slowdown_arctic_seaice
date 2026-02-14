# -----------------------------
# LOAD + RE-PLOT FROM NETCDF
# -----------------------------
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------
#------------------------------------------------------
# ROOT PATH
rootpath = '/cofast/lhoffman/slowdown/'
#------------------------------------------------------
#------------------------------------------------------

# ---------- CONFIG ----------
# Map each exp_number to its own file (edit these paths for your setup)
exp_numbers = [1]
loadpaths = {
    1: rootpath+"M1_EI.X_performance_TVT_multiRun_seasonal_globalSTD.nc",

}

# custom legend labels for each exp_number
exp_labels_custom = {
    1: "full",

}

exp_labels_name = [
    "JJA",
    "MAM",
    "DJF",
    "SON",
    "JJA, MAM, DJF, SON",

    
]
x = np.arange(len(exp_labels_name))  # positions along x-axis

pretty_names = {
    "AUPRC": "AUPRC",
    "AUROC": "AUROC",
    "Prec@Thr": "Precision (at thr)",
    "Rec@Thr": "Recall (at thr)",
    "F1": "F1 (at thr)",
    "Accuracy": "Accuracy",
    "MCC": "MCC",
    "Brier": "Brier score",
    "BinaryCrossEntropy": "Binary CE",
}

# ---------- LOAD ----------
ds_by_num = {n: xr.open_dataset(p) for n, p in loadpaths.items()}

# Use labels from any one file (assumed consistent)
exp_labels = ds_by_num[exp_numbers[0]]["experiment"].values
metrics = ds_by_num[exp_numbers[0]]["metric"].values
splits  = ds_by_num[exp_numbers[0]]["split"].values

# x positions for experiments; tick labels are the experiment names
x = np.arange(len(exp_labels))

# colors per exp_number (distinct lines)
line_colors = {
    1: "tab:blue",

}

# grid layout for metrics
n_m = len(metrics)
ncols = 3
nrows = int(np.ceil(n_m / ncols))

# ---------- PLOT: one figure per split ----------
for split in splits:
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.2 * nrows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for ax, (m_idx, m) in zip(axes, enumerate(metrics)):
        for n in exp_numbers:
            ds = ds_by_num[n]

            y_da  = ds["metric_value"].sel(metric=m, split=split).mean("run")
            lo_da = ds["ci_low"].sel(metric=m, split=split)
            hi_da = ds["ci_high"].sel(metric=m, split=split)

            y  = y_da.sel(experiment=exp_labels).values
            lo = lo_da.sel(experiment=exp_labels).values
            hi = hi_da.sel(experiment=exp_labels).values

            yerr = np.vstack([y - lo, hi - y])  
            ax.errorbar(
                x, y, yerr=yerr, fmt="o-", capsize=3,
                label=exp_labels_custom[n],
                color=line_colors.get(n, None)
            )

        ax.set_title(pretty_names.get(m.item() if hasattr(m, "item") else m, str(m)))
        ax.set_ylabel(pretty_names.get(m.item() if hasattr(m, "item") else m, str(m)))
        ax.grid(True, linestyle="--", alpha=0.4)

    # remove unused axes
    for j in range(n_m, axes.size):
        fig.delaxes(axes[j])

    for ax in axes[:n_m]:
        ax.set_xticks(x)
        ax.set_xticklabels(exp_labels_name, rotation=45, ha="right")

    # ---- figure-level legend above all subplots ----
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(exp_numbers),
        frameon=False, bbox_to_anchor=(0.5, 1.05)
    )

    fig.suptitle(f"Performance by exp_type — split: {split}", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()