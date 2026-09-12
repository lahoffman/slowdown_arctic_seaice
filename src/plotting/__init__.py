"""
src/plotting — figure code, kept separate from analysis modules.

Modules
-------
style        : shared colours / colormaps, paper_rc, tidy, save, panel_label.
slowdowns    : time series with trend segments, member highlights, label
               distributions, SIE-vs-GMT panels, relative-label diagnostics.
maps         : global SST / relevance composites, region boxes (cartopy optional).
performance  : PR curve, confusion matrices, metric strip, member timeline.
conditional  : P(event | climate-index phase) bars.
observations : CNN votes on observed SST with index panels (Fig. 4).
baselines    : baseline-vs-CNN skill strip plot.
paper        : manuscript figures, one function per figure; built from cached
               outputs by scripts/make_figure.py.

Import submodules directly (``from src.plotting import slowdowns``); each
sets the Agg backend so scripts run headless.
"""
