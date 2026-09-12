"""
performance.py — CNN skill panels (Figs. S3–S6).
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve

from . import style as st
from .style import plt


def learning_curve(ax, history: Dict[str, Sequence[float]]) -> None:
    ep = np.arange(1, len(history["loss"]) + 1)
    ax.plot(ep, history["loss"], color=st.BLUE, lw=1.8, label="train loss")
    ax.plot(ep, history["val_loss"], color=st.BLUE, lw=1.8, ls="--", label="val loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(frameon=False)
    st.tidy(ax)


def pr_curve(ax, y_true, y_score) -> float:
    """Precision, recall and F1 vs threshold; returns the selected threshold."""
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)
    thr = t[np.argmin(np.abs(p[:-1] - r[:-1]))]
    ax.plot(t, p[:-1], color=st.BLUE, lw=1.8, label="Precision")
    ax.plot(t, r[:-1], color=st.ORANGE, lw=1.8, label="Recall")
    ax.plot(t, f1, color=st.GREEN, lw=1.8, label="F1")
    ax.axvline(thr, color=st.INK, lw=1.5, label=f"threshold = {thr:.3f}")
    ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.legend(frameon=False)
    st.tidy(ax)
    return float(thr)


def confusion_panel(ax, y_true, y_score, threshold: float, title: str = "") -> None:
    cm = confusion_matrix(y_true, (y_score >= threshold).astype(int))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else st.INK)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative", "Positive"]); ax.set_yticklabels(["Negative", "Positive"])
    ax.set_xlabel("predicted label"); ax.set_ylabel("true label")
    ax.set_title(title, loc="left", weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def metric_strip(ax, values: Dict[str, np.ndarray], as_percent: bool = True) -> None:
    """One column per metric: all split×seed values (dots) and their mean (bar)."""
    names = list(values)
    for i, k in enumerate(names):
        v = np.asarray(values[k]).ravel()
        v = v[np.isfinite(v)]
        ax.scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.15, 0.15, v.size),
                   v, color=st.BLUE, alpha=0.5, s=18, zorder=2)
        ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color="#8b1a1a", lw=2.5, zorder=3)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    if as_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}"))
        ax.set_ylabel("score (%)")
    else:
        ax.set_ylabel("score")
    st.tidy(ax)


def member_timeline(ax, y_true, y_pred, years, member_labels) -> None:
    """Actual vs predicted slowdowns per test member (rows) over onset year."""
    nm = len(member_labels)
    yt, yp = np.asarray(y_true).reshape(nm, -1), np.asarray(y_pred).reshape(nm, -1)
    for i, m in enumerate(member_labels):
        ax.axhline(m, color=st.GRID, lw=0.6, zorder=0)
        a = years[yt[i] == 1]
        ax.scatter(a, np.full(a.size, m), color="dimgray", s=40, zorder=2,
                   label="actual slowdowns" if i == 0 else "")
        tp = years[(yp[i] == 1) & (yt[i] == 1)]
        ax.scatter(tp, np.full(tp.size, m), color=st.BLUE, s=16, zorder=3,
                   label="correct predictions" if i == 0 else "")
        fp = years[(yp[i] == 1) & (yt[i] == 0)]
        ax.scatter(fp, np.full(fp.size, m), color=st.ORANGE, s=16, zorder=3,
                   label="incorrect predictions" if i == 0 else "")
    ax.set_xlabel("onset year"); ax.set_ylabel("ensemble member")
    ax.set_yticks(member_labels); ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), frameon=False)
    st.tidy(ax, grid_axis="x")
