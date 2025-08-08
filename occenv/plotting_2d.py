import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---- These functions are used in plot_pmf.py ----
def plot_pmf_with_normals(
    numbers_covered: list[int],
    union_pmf: list[float],
    x_continuous: list[float],
    mu_union,
    sigma_union,
    mu_union_approx,
    sd_union_approx,
    title,
    save_path=None,
):
    """
    Plot discrete union PMF with analytical (solid blue) and approximated (dashed red) normal overlays.
    """
    normal_union = norm.pdf(x_continuous, mu_union, sigma_union)
    normal_union_approx = norm.pdf(x_continuous, mu_union_approx, sd_union_approx)

    plt.figure(figsize=(9, 6))
    plt.bar(numbers_covered, union_pmf, alpha=0.6, label="Probability Mass Function")
    plt.plot(x_continuous, normal_union, "b", label="Analytical normal distribution")
    plt.plot(
        x_continuous,
        normal_union_approx,
        "r--",
        label="Approximated normal distribution",
    )
    plt.legend()
    plt.xlabel("Total union size ($N'$)")
    plt.ylabel("$P(X=N')$")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_line_graph(x_values, pmf, title, xlabel, save_path=None):
    plt.plot(x_values, pmf, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("P(X ≥ x)")
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
    title,
    xlabel,
    vlines: list[tuple[float, str, str, str]],
    bins=300,
    save_path=None,
):
    """
    Histogram for a PMF with normal overlay
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
        plt.plot(x_grid, norm_pdf, label=f"normal pdf (μ={mu:.2f}, σ={sigma:.2f})")

    # vertical reference lines
    for x, lbl, color, linestyle in vlines:
        plt.axvline(x=x, color=color, linestyle=linestyle, label=lbl)

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
