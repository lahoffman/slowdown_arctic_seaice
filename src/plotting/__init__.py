"""
src/plotting — figure code, kept separate from analysis modules.

Modules
-------
style      : shared colours, axis styling, save helper.
slowdowns  : relative-label diagnostics (per-file 3-panel, window sweep).
baselines  : baseline-vs-CNN skill strip plot.

Import submodules directly (``from src.plotting import slowdowns``); each
sets the Agg backend so scripts run headless.
"""
