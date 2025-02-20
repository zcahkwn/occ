from occenv.constants import FIGURE_DIR
from occenv.analytical import Analytical_result
import numpy as np
import matplotlib.pyplot as plt


def create_meshgrid(start=1):
    n_vals = np.arange(start, N)
    return np.meshgrid(n_vals, n_vals)


def create_heatmap(x, y, z, xlabel, ylabel, title, filename, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(x, y, z, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.savefig(FIGURE_DIR / filename)
    plt.close(fig)


def plot_collude_2():
    x, y = create_meshgrid()
    z = np.vectorize(
        lambda n1, n2: (
            Analytical_result(N, [n1, n2]).collude_prob(N) if n1 + n2 >= N else np.nan
        )
    )(x, y)
    create_heatmap(
        x,
        y,
        z,
        "$n_1$",
        "$n_2$",
        f"Probability of Collusion with $N={N},m=2$",
        f"collude_2_{N}.pdf",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )


def plot_sigma_2():
    x, y = create_meshgrid()
    z = np.vectorize(lambda n1, n2: Analytical_result(N, [n1, n2]).compute_sigma())(
        x, y
    )
    # z = (N*(x+y)-x*y)/(N**2)
    create_heatmap(
        x,
        y,
        z,
        "$n_1$",
        "$n_2$",
        f"$\sigma$ with $N={N},m=2$",
        f"sigma_2_{N}.pdf",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )


def plot_occ_2():
    x, y = create_meshgrid()
    z = np.vectorize(lambda n1, n2: Analytical_result(N, [n1, n2]).occ_value())(x, y)
    create_heatmap(
        x,
        y,
        z,
        "$n_1$",
        "$n_2$",
        f"OCC with $N={N},m=2$",
        f"occ_2_{N}.pdf",
        cmap="Blues",
        vmin=0.1,
        vmax=1,
    )


def plot_occ_relation_2():
    x, y = create_meshgrid()
    z = np.vectorize(
        lambda n1, n2: Analytical_result(N, [n1, n2]).collude_prob(N)
        / Analytical_result(N, [n1, n2]).occ_value()
    )(x, y)
    create_heatmap(
        x,
        y,
        z,
        "$n_1$",
        "$n_2$",
        f"Ratio between OCC and Collusion with $N={N},m=2$",
        f"relation_occ_2_{N}.pdf",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )


def plot_sigma_relation_2():
    x, y = create_meshgrid()
    z = np.vectorize(
        lambda n1, n2: Analytical_result(N, [n1, n2]).collude_prob(N)
        / Analytical_result(N, [n1, n2]).compute_sigma()
    )(x, y)
    create_heatmap(
        x,
        y,
        z,
        "$n_1$",
        "$n_2$",
        f"Ratio between sigma and Collusion with $N={N},m=2$",
        f"relation_sigma_2_{N}.pdf",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )


if __name__ == "__main__":
    N = 10
    plot_collude_2()
    plot_sigma_2()
    plot_occ_2()
    plot_occ_relation_2()
    plot_sigma_relation_2()
