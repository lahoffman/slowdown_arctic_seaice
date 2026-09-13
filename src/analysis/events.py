"""
events.py — slowdown *events* (runs of consecutive positive onset years per
member) and event-level scoring (steps 3.1, 3.3).

A 10-yr trend flagged in onset years 2003, 2004, 2005 is one event, not three
independent samples. Event-level hit rate = fraction of events with at least
one predicted-positive year; false-alarm runs = predicted runs overlapping no event.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def runs(binary: np.ndarray) -> List[Tuple[int, int]]:
    """(start, end) index pairs of consecutive 1s in a 1-D 0/1 array."""
    b = np.asarray(binary).astype(int)
    if b.size == 0:
        return []
    d = np.diff(np.concatenate([[0], b, [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1))


def event_table(labels: np.ndarray, years: np.ndarray) -> Dict[str, np.ndarray]:
    """Events for (nens, nyear) labels: member, onset year, duration (yr) of each run."""
    mem, y0, dur = [], [], []
    for m in range(labels.shape[0]):
        for s, e in runs(labels[m]):
            mem.append(m); y0.append(int(years[s])); dur.append(e - s + 1)
    return {"member": np.array(mem), "onset": np.array(y0), "duration": np.array(dur)}


def summarize_events(labels: np.ndarray, years: np.ndarray) -> Dict[str, float]:
    ev = event_table(labels, years)
    return {"positive_years": int(labels.sum()), "events": int(ev["duration"].size),
            "mean_duration": float(ev["duration"].mean()) if ev["duration"].size else np.nan,
            "median_duration": float(np.median(ev["duration"])) if ev["duration"].size else np.nan,
            "max_duration": int(ev["duration"].max()) if ev["duration"].size else 0,
            "events_per_member": float(ev["duration"].size / labels.shape[0]),
            "sample_to_event_ratio": float(labels.sum() / max(ev["duration"].size, 1))}


def event_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Event-level scores for one member's (nyear,) truth and binary prediction:
    hit = event with ≥1 predicted-positive year; false-alarm run = predicted run
    overlapping no true event.
    """
    true_runs, pred_runs = runs(y_true), runs(y_pred)
    hits = sum(any(ps <= e and pe >= s for ps, pe in pred_runs) for s, e in true_runs)
    fa = sum(not any(s <= pe and e >= ps for s, e in true_runs) for ps, pe in pred_runs)
    return {"events": len(true_runs), "hits": hits, "pred_runs": len(pred_runs), "false_alarm_runs": fa}


def event_scores_members(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Aggregate ``event_scores`` over the member axis of (nmem, nyear) arrays."""
    tot = {"events": 0, "hits": 0, "pred_runs": 0, "false_alarm_runs": 0}
    for m in range(y_true.shape[0]):
        for k, v in event_scores(y_true[m], y_pred[m]).items():
            tot[k] += v
    tot["hit_rate"] = tot["hits"] / tot["events"] if tot["events"] else np.nan
    tot["false_alarm_ratio"] = tot["false_alarm_runs"] / tot["pred_runs"] if tot["pred_runs"] else np.nan
    return tot
