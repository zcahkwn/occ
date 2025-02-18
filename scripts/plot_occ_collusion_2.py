import numpy as np
import matplotlib.pyplot as plt
from math import comb, sqrt, log, exp
# from mpl_toolkits.mplot3d import Axes3D  
from occenv.constants import FIGURE_DIR 
from scripts.collusion_prob import collusion_2

def create_meshgrid(start=1, stop=100):
    n_vals = np.arange(start, stop)
    return np.meshgrid(n_vals, n_vals)

def create_3d_plot(x, y, z, xlabel, ylabel, zlabel, title, filename):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)
    plt.savefig(FIGURE_DIR / filename)
    plt.close(fig)

def creat_heatmap(x, y, z, xlabel, ylabel, title, filename, cmap, vmin, vmax):
    fig, ax = plt.subplots(figsize=(8, 6))
    c = ax.pcolormesh(x, y, z, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    plt.savefig(FIGURE_DIR / filename)
    plt.close(fig)


def plot_collude_2():
    N = 100
    x, y = create_meshgrid()
    z = np.vectorize(lambda n1, n2: collusion_2(N, [n1, n2]))(x, y)
    # create_3d_plot(
    #     x, y, z,
    #     '$n_1$', '$n_2$', 'Probability of Collusion',
    #     'Probability of Collusion (m=2) with $N=10$', "collude_2.pdf"
    # )
    creat_heatmap(
        x, y, z,
        '$n_1$', '$n_2$',
        'Probability of Collusion (m=2) with $N=100$', "collude_2_100.pdf", cmap='Blues', vmin=0, vmax=1
    )

def plot_occ_2():
    N = 100
    x, y = create_meshgrid()
    z = np.sqrt(x * y) / N
    # create_3d_plot(
    #     x, y, z,
    #     '$n_1$', '$n_2$', r'OCC when m=2',
    #     r'3D Surface Plot of OCC when m=2', "occ_2.pdf"
    # )
    creat_heatmap(
        x, y, z,
        '$n_1$', '$n_2$', r'OCC (m=2) with $N=100$', "occ_2_100.pdf", cmap='Blues', vmin=0, vmax=1
    )

def plot_relation_2():
    N=100
    x, y = create_meshgrid()
    z=np.vectorize(lambda n1, n2: collusion_2(N, [n1, n2])/(n1*n2/N**2) if (n1+n2) >= N else np.nan)(x, y)
    creat_heatmap(
        x, y, z,
        '$n_1$', '$n_2$', r'Ratio between OCC and Collusion (m=2)', "relation_2_100.pdf", cmap='viridis', vmin=0, vmax=1
    )

if __name__ == "__main__":
    plot_collude_2()
    plot_occ_2()
    plot_relation_2()