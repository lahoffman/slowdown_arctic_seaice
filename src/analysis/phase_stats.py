"""
phase_stats.py — event dependence on climate-index phase.

Phase labellers turn a JJA index into −1/0/+1 categories; the statistics
give P(event | phase) with bootstrap CIs and the variance explained by
phase (law of total variance). Used by Fig. 3 / S11.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy.stats import chi2_contingency

#: name → (labeller, phases, phase names)
PHASE_SPECS = {
    "arctic": (lambda v: _threshold_labels(v, 1.0), (-1, 0, 1), ("−", "Neutral", "+")),
    "ipo":    (lambda v: np.where(np.asarray(v) <= 0, -1, 1).astype(int), (-1, 1), ("−", "+")),
    "nino34": (lambda v: _threshold_labels(v, 0.4), (-1, 0, 1), ("La Niña", "Neutral", "El Niño")),
}
INDEX_TITLES = {"arctic": "Arctic SST", "ipo": "IPO", "nino34": "Niño3.4"}


def _threshold_labels(vals, thr: float) -> np.ndarray:
    vals = np.asarray(vals, float)
    out = np.zeros(vals.size, int)
    out[vals <= -thr] = -1
    out[vals >= thr] = 1
    return out


def event_dependence(phase: np.ndarray, event: np.ndarray, phases: Sequence[int],
                     n_boot: int = 5000, seed: int = 42) -> Dict:
    """P(event | phase) with bootstrap CIs and a chi² p-value."""
    phase, event = np.asarray(phase), np.asarray(event, bool)
    rng = np.random.default_rng(seed)
    probs, lo, hi, n_phase = {}, {}, {}, {}
    for ph in phases:
        sub = event[phase == ph].astype(float)
        n_phase[ph] = int(sub.size)
        if sub.size == 0:
            probs[ph] = lo[ph] = hi[ph] = np.nan
            continue
        probs[ph] = sub.mean()
        boots = rng.choice(sub, size=(n_boot, sub.size), replace=True).mean(1)
        lo[ph], hi[ph] = np.percentile(boots, [2.5, 97.5])
    table = np.array([[event[phase == ph].sum(), (phase == ph).sum() - event[phase == ph].sum()]
                      for ph in phases])
    chi2_p = chi2_contingency(table)[1] if (table.sum(1) > 0).all() else np.nan
    return dict(probs=probs, ci_lo=lo, ci_hi=hi, n_phase=n_phase, chi2_p=chi2_p)


def event_dependence_multiseed(phase: np.ndarray, events: List[np.ndarray],
                               phases: Sequence[int], n_boot: int = 2000,
                               seed: int = 42) -> Dict:
    """P(event | phase) averaged over seeds, nested bootstrap (seeds × samples)."""
    phase = np.asarray(phase)
    events = [np.asarray(e, bool) for e in events]
    rng = np.random.default_rng(seed)
    probs = {ph: np.nanmean([e[phase == ph].mean() for e in events]) for ph in phases}
    lo, hi = {}, {}
    for ph in phases:
        sel = phase == ph
        subs = [e[sel].astype(float) for e in events]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            ids = rng.integers(0, len(events), len(events))
            boots[b] = np.mean([rng.choice(subs[i], size=subs[i].size, replace=True).mean()
                                for i in ids])
        lo[ph], hi[ph] = np.percentile(boots, [2.5, 97.5])
    return dict(probs=probs, ci_lo=lo, ci_hi=hi)


def variance_explained(phase: np.ndarray, event: np.ndarray, phases: Sequence[int],
                       n_boot: int = 5000, seed: int = 42) -> Dict:
    """Between-phase / total variance of the binary event, with bootstrap CI."""
    phase, event = np.asarray(phase), np.asarray(event, float)

    def ve(ph_arr, ev_arr):
        tv = ev_arr.var()
        if tv == 0:
            return np.nan
        gm = ev_arr.mean()
        bw = sum((ph_arr == ph).sum() * (ev_arr[ph_arr == ph].mean() - gm) ** 2
                 for ph in phases if (ph_arr == ph).any())
        return bw / ev_arr.size / tv

    rng = np.random.default_rng(seed)
    n = event.size
    boots = np.array([ve(phase[i], event[i]) for i in rng.integers(0, n, (n_boot, n))])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(ve=ve(phase, event), ci_lo=lo, ci_hi=hi)


def phase_summary(index_vals: np.ndarray, slow: np.ndarray, tp_seeds: List[np.ndarray],
                  key: str, n_boot: int = 5000) -> Dict:
    """Everything one Fig. 3 panel needs for index ``key``."""
    labeller, phases, names = PHASE_SPECS[key]
    ph = labeller(index_vals)
    out = dict(key=key, title=INDEX_TITLES[key], phases=phases, names=names,
               all=event_dependence(ph, slow, phases, n_boot),
               ve_all=variance_explained(ph, slow, phases, n_boot))
    if tp_seeds:
        out["tp"] = event_dependence_multiseed(ph, tp_seeds, phases, min(n_boot, 2000))
        tp_avg = np.mean(np.stack(tp_seeds), 0) >= 0.5
        out["ve_tp"] = variance_explained(ph, tp_avg, phases, n_boot)
    return out
