"""
src/plotting — figure code, kept separate from analysis modules.

Modules
-------
style      : shared colours, axis styling, save helper.
slowdowns  : relative-label diagnostics (per-file 3-panel, window sweep).
baselines  : baseline-vs-CNN skill strip plot.
paper      : manuscript figures, one function per figure (fig_s1, ...);
             built from cached outputs by scripts/make_figure.py.

Import submodules directly (``from src.plotting import slowdowns``); each
sets the Agg backend so scripts run headless.
"""
