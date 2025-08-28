import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import chi2


class Gaussian2D:
    """
    Convenience wrapper for a 2D Gaussian, after mu and Sigma are calculated.
    Doesn't matter if the original distribution is discrete or continuous.
    """

    def __init__(
        self,
        mu: list[float, float],
        Sigma: list[list[float, float], list[float, float]],
    ):
        self.mu = np.array(mu, float).reshape(2)  # mean vector
        self.Sigma = np.array(Sigma, float).reshape(2, 2)  # covariance matrix
        self._Sinv = np.linalg.inv(self.Sigma)  # inverse of the covariance matrix
        self._det = float(
            np.linalg.det(self.Sigma)
        )  # determinant of the covariance matrix
        self._chol = None  # Cholesky decomposition

        # Compute eigendecomposition of the covariance matrix
        evals, evecs = np.linalg.eigh(self.Sigma)
        order = evals.argsort()[::-1]
        self.evals = evals[order]  # eigenvalues (largest first)
        self.evecs = evecs[:, order]  # eigenvectors (columns, largest eigenvalue first)

    @property
    def Sinv(self):
        return self._Sinv

    @property
    def det(self):
        return self._det

    @property
    def chol(self):
        if self._chol is None:
            self._chol = np.linalg.cholesky(
                self.Sigma
            )  # calculate cholesky decomposition
        return self._chol

    def mahalanobis2(self, X):
        """Return M^2 for points X (...,2). M² = (x - μ)ᵀ Σ⁻¹ (x - μ)"""
        X = np.asarray(X, float)
        dX = X - self.mu
        return np.einsum("...i,ij,...j->...", dX, self.Sinv, dX)

    def pdf(self, X):
        """Unnormalized-safe 2D Gaussian pdf (continuous) at points X (...,2)."""
        M2 = self.mahalanobis2(X)
        const = 1.0 / (2.0 * np.pi * np.sqrt(self.det))
        return const * np.exp(
            -0.5 * M2
        )  # this is the probability density function of the bivariate Gaussian

    def whiten(self, X):
        """
        Perform whitening transformation, so mean becomes 0 and covariance becomes identity matrix.
        z = L^{-1}(x - mu) with LL^T = Sigma."""
        X = np.asarray(X, float)
        return np.linalg.solve(self.chol, (X - self.mu).T).T

    def ellipse(self, ax, p=0.95, **kw):
        """Add a confidence ellipse for mass p.
        The confidence ellipse is the region that contains p of the data.
        """
        c = float(
            chi2.ppf(p, df=2)
        )  # confidence level for a chi-squared distribution with 2 degrees of freedom

        width, height = 2 * np.sqrt(
            c * self.evals
        )  # width and height of the ellipse are scaled by eigenvalues and confidence level

        angle = np.degrees(
            np.arctan2(self.evecs[1, 0], self.evecs[0, 0])
        )  # angle of the ellipse is the angle of the eigenvector

        e = Ellipse(
            xy=self.mu, width=width, height=height, angle=angle, fill=False, **kw
        )
        ax.add_patch(e)
        return e
