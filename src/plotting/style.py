"""
style.py — shared colours and helpers for project figures.

Categorical colours follow the Okabe–Ito colour-blind-safe set.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#1f2933", "#8a949e", "#e5e8eb"
BLUE, ORANGE, GREEN, PINK, AMBER, SKY = ("#0072B2", "#D55E00", "#009E73",
                                         "#CC79A7", "#E69F00", "#56B4E9")
CATEGORICAL = [BLUE, ORANGE, GREEN, PINK, AMBER, SKY]

#: Semantic colours reused across figures.
C_ALL, C_CMIP6, C_SMBB, C_ORIG, C_EVENT = BLUE, ORANGE, GREEN, "#999999", AMBER


def tidy(ax, grid_axis: str = "y") -> None:
    """Recessive grid, no top/right spines."""
    ax.grid(axis=grid_axis, color=GRID, lw=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, out_png, dpi: int = 200) -> None:
    """Save, close, and report the path."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  figure → {out_png}")
