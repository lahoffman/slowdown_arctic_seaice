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
C_SLOW, C_NOSLOW = BLUE, ORANGE          # slowdown / non-slowdown (was green / red)
C_MEMBER = "#5e4fa2"                      # highlighted ensemble member
C_THR_OBS, C_THR_MODEL = SKY, "#a6cee3"   # observed / model thresholds
C_POS, C_NEG = ORANGE, BLUE               # positive / negative index phase shading

#: Colormaps for maps (cmocean.balance if available, else RdBu_r).
try:
    import cmocean  # noqa: F401
    CMAP_SST = cmocean.cm.balance
except ImportError:  # pragma: no cover
    CMAP_SST = "RdBu_r"
CMAP_LRP_SIGNED, CMAP_LRP_POS = "PuOr_r", "magma"


def paper_rc(base: float = 14) -> None:
    """Set font sizes for manuscript figures (call once per script). Legends/labels follow these."""
    plt.rcParams.update({"font.size": base, "axes.titlesize": base + 1,
                         "axes.labelsize": base, "xtick.labelsize": base - 1,
                         "ytick.labelsize": base - 1, "legend.fontsize": base - 2,
                         "figure.titlesize": base + 1, "lines.linewidth": 1.6})


def panel_label(ax, label: str, size=None, x: float = 0.0, y: float = 1.03) -> None:
    """Bold (a)/(b)/… label above the top-left corner of an axis (size follows axes.titlesize)."""
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size or plt.rcParams["axes.titlesize"],
            fontweight="bold", va="bottom", ha="left")


def tidy(ax, grid_axis: str = "y") -> None:
    """No grid, no top/right spines (``grid_axis`` kept for call compatibility)."""
    ax.grid(False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def save(fig, out_png, dpi: int = 200) -> None:
    """Save, close, and report the path."""
    import warnings
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)   # shared-colorbar grids
        fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  figure → {out_png}")
