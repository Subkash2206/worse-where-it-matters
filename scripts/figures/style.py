"""Shared matplotlib style for the paper figures.

Every figure script imports this module, calls :func:`apply` once and :func:`save` at the end, so
the whole figure set is consistent and re-runnable after new results land.  Outputs are written to
``results/figures_paper/`` as vector PDF **and** PNG at 300 dpi (the PDF for the manuscript, the PNG
for slides and quick review).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display, no GPU, safe while training jobs own the machine

import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "results" / "figures_paper"
RESULTS = REPO_ROOT / "results"

#: one colour per setting; the two internal architectures share the blue family
PALETTE = {
    "cvc_unet": "#1f4e79",
    "cvc_swin": "#2e86c1",
    "brats": "#7d3c98",
    "kvasir_seg": "#c0392b",
    "cvc_colondb": "#e67e22",
    "etis_larib": "#16a085",
}
MARKERS = {
    "cvc_unet": "o",
    "cvc_swin": "s",
    "brats": "D",
    "kvasir_seg": "^",
    "cvc_colondb": "v",
    "etis_larib": "P",
}
LABELS = {
    "cvc_unet": "CVC-ClinicDB, UNet",
    "cvc_swin": "CVC-ClinicDB, SwinUNETR",
    "brats": "BraTS 2021, SegResNet",
    "kvasir_seg": "Kvasir-SEG (external)",
    "cvc_colondb": "CVC-ColonDB (external)",
    "etis_larib": "ETIS-Larib (external)",
}


def apply() -> None:
    """Consistent, publication-ready defaults (single-column width, no chart junk)."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.35,
            "grid.color": "#b0b0b0",
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.4,
            "lines.markersize": 4,
            "errorbar.capsize": 2,
            "pdf.fonttype": 42,  # editable text in the PDF, no font substitution
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> list[Path]:
    """Write ``name`` as PDF and PNG into ``results/figures_paper`` and return the paths."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "png"):
        path = FIGURE_DIR / f"{name}.{suffix}"
        fig.savefig(path, dpi=300)
        paths.append(path)
    plt.close(fig)
    for path in paths:
        print(f"wrote {path}")
    return paths


def annotate(ax, text: str, *, corner: str = "lower right") -> None:
    """Small panel label (a), (b), ... or a note, in a fixed corner."""
    x, y = (0.98, 0.02) if "right" in corner else (0.02, 0.02)
    if "upper" in corner:
        y = 0.98
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right" if "right" in corner else "left",
        va="top" if "upper" in corner else "bottom",
        fontsize=6.5,
        color="#404040",
    )
