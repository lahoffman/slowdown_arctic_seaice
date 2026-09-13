"""
09_event_stats.py — independent events and event-level skill (steps 3.1, 3.3).

(1) How many *events* (runs of consecutive flagged onset years per member) the
    labels contain, vs the number of positive member-years the metrics count.
(2) Event-level hit rate and false-alarm ratio of a CNN configuration's cached
    test predictions (--tag), alongside the sample-level F1.

Outputs: results/events[/<tag>]/event_summary.md, FIGURES_DIR/diagnostics/event_stats[_<tag>].png

Usage:
  python scripts/09_event_stats.py                 # labels only
  python scripts/09_event_stats.py --tag rel_base  # + event-level CNN scores
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs import paths
from src.analysis import baselines as bl, events as ev


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-file", type=Path, default=paths.CESM2LE_SLOWDOWNS_DIR /
                   "cesm2le_sie_slowdown_relative_SEP_w10_s1_group_1990-2100.nc")
    p.add_argument("--original-file", type=Path, default=None, help="v1 labels for comparison (default: 02 output)")
    p.add_argument("--tag", default=None)
    p.add_argument("--start-year", type=int, default=1990); p.add_argument("--end-year", type=int, default=2030)
    p.add_argument("--no-fig", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    out_dir = paths.RESULTS_DIR / "events" / a.tag if a.tag else paths.RESULTS_DIR / "events"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels, years = bl.load_labels(a.labels_file, a.start_year, a.end_year)
    orig_file = a.original_file or paths.cesm2le_slowdown_file("sie", "SEP")
    rows = {"relative (1990–2030)": ev.summarize_events(labels, years)}
    if orig_file.exists():
        lab_o, yrs_o = bl.load_labels(orig_file, a.start_year, 2040)
        rows["original (1990–2040)"] = ev.summarize_events(lab_o, yrs_o)
    keys = ["positive_years", "events", "mean_duration", "median_duration", "max_duration",
            "events_per_member", "sample_to_event_ratio"]
    lines = ["| labels | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]) for k in keys) + " |")
    md = "# Slowdown events (runs of consecutive flagged onset years per member)\n\n" + "\n".join(lines) + "\n"

    cnn_rows = []
    if a.tag:
        nyear = years.size
        for k, te_b, va_b, tr_b in bl.iter_split_blocks(9, 10):
            preds = bl.load_cnn_test_predictions(paths.cesm2le_predictions_dir(a.tag), k)
            for r, (yt, yp, thr) in enumerate(preds):
                if yt.size != 10 * nyear:
                    continue
                yt2, pr2 = yt.reshape(10, nyear), (yp >= thr).astype(int).reshape(10, nyear)
                s = ev.event_scores_members(yt2, pr2); s.update(split=k, run=r)
                tp = (yt2 * pr2).sum(); s["f1_sample"] = 2 * tp / max(yt2.sum() + pr2.sum(), 1)
                cnn_rows.append(s)
        if cnn_rows:
            hr = np.array([r["hit_rate"] for r in cnn_rows]); far = np.array([r["false_alarm_ratio"] for r in cnn_rows])
            f1 = np.array([r["f1_sample"] for r in cnn_rows])
            md += (f"\n# Event-level skill, tag `{a.tag}` ({len(cnn_rows)} split×seed models, test members)\n\n"
                   f"| | median | range |\n|---|---|---|\n"
                   f"| event hit rate (≥1 predicted year in the event) | {np.median(hr):.2f} | {hr.min():.2f}–{hr.max():.2f} |\n"
                   f"| false-alarm ratio (predicted runs touching no event) | {np.median(far):.2f} | {far.min():.2f}–{far.max():.2f} |\n"
                   f"| sample-level F1 (same models) | {np.median(f1):.2f} | {f1.min():.2f}–{f1.max():.2f} |\n")
    (out_dir / "event_summary.md").write_text(md); print(md)
    if not a.no_fig:
        from src.plotting import sensitivity as plot, style as st
        st.paper_rc()
        plot.plot_events(ev.event_table(labels, years), cnn_rows,
                         paths.FIGURES_DIR / "diagnostics" / f"event_stats{'_' + a.tag if a.tag else ''}.png")


if __name__ == "__main__":
    main()
