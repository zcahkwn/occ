"""
This module contains functions for plotting 2D plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---- These functions are used in plot_univariate_pmf.py ----
def plot_pmf_with_normals(
    x_values: list[int],
    pmf: list[float],
    x_continuous: list[float],
    mu,
    sigma,
    mu_approx,
    sd_approx,
    xlabel,
    title,
    save_path=None,
):
    """
    Plot discrete union PMF with analytical (solid blue) and approximated (dashed red) normal overlays.
    """
    normal = norm.pdf(x_continuous, mu, sigma)
    normal_approx = norm.pdf(x_continuous, mu_approx, sd_approx)

    plt.figure(figsize=(9, 6))
    plt.bar(x_values, pmf, alpha=0.6, label="Probability Mass Function")
    plt.plot(x_continuous, normal, "b", label="Analytical normal distribution")
    plt.plot(
        x_continuous,
        normal_approx,
        "r--",
        label="Approximated normal distribution",
    )
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(f"$P(X={xlabel})$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_line_graph(x_values, pmf, title, xlabel, save_path=None):
    plt.plot(x_values, pmf, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(f"$P(X ≥ {xlabel})$")
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


# ---- These functions are used in plot_jaccard.py ----
def plot_hist_with_normal(
    x_values,
    pmf,
    mu,
    sigma,
    mu_approx,
    sigma_approx,
    title,
    xlabel,
    bins=300,
    save_path=None,
):
    """
    Histogram for a PMF with analytical (solid blue) and approximated (dashed red) normal overlays
    """
    plt.figure(figsize=(9, 4))
    plt.hist(
        x_values,
        bins=np.linspace(0, max(x_values) + 1e-6, bins),
        weights=pmf,
        density=True,
        alpha=0.4,
        label="empirical pmf",
    )

    # normal PDF overlay
    x_grid = np.linspace(0, max(x_values), 400)
    if sigma > 0:
        norm_pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -((x_grid - mu) ** 2) / (2 * sigma**2)
        )
        plt.plot(
            x_grid,
            norm_pdf,
            label=f"normal pdf (μ={mu:.4f}, σ={sigma:.4f})",
            color="b",
            linestyle="-",
        )

    # approximated normal PDF overlay
    if sigma_approx > 0:
        norm_pdf_approx = (1 / (sigma_approx * np.sqrt(2 * np.pi))) * np.exp(
            -((x_grid - mu_approx) ** 2) / (2 * sigma_approx**2)
        )
        plt.plot(
            x_grid,
            norm_pdf_approx,
            label=f"approximated normal pdf (μ={mu_approx:.4f}, σ={sigma_approx:.4f})",
            color="r",
            linestyle="--",
        )

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_stem_pmf(
    x_values: list[float],
    pmf: list[float],
    title: str,
    xlabel: str,
    vlines: list[tuple[float, str, str, str]],
    save_path: str | None = None,
) -> None:
    """
    Stem plot for discrete PMF
    """
    plt.figure(figsize=(9, 4))
    plt.stem(x_values, pmf)

    for x, lbl, color, linestyle in vlines:
        plt.axvline(x=x, color=color, linestyle=linestyle, label=lbl)

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
