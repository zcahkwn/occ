import numpy as np
from math import comb, prod
from matplotlib.patches import Ellipse

# __all__ = [
#     "Gaussian2D",
#     "BivariateGrid",
#     "gaussian_from_grid",
#     "gaussian_from_grid_new",
#     "draw_cov_ellipse",
#     "compute_grid",
#     "gaussian_on_grid",
#     "compare_distributions",
#     "mahalanobis_ks",
#     "mardia_moments",
#     "angle_uniformity",
#     "conditional_linearity",
#     "mardia_skewness",
# ]


# ----------------------------
# Core classes (minimal set)
# ----------------------------
class Gaussian2D:
    """Convenience wrapper for a 2D Gaussian."""

    def __init__(self, mu, Sigma):
        self.mu = np.asarray(mu, float).reshape(2)
        self.Sigma = np.asarray(Sigma, float).reshape(2, 2)
        self._Sinv = np.linalg.inv(self.Sigma)
        self._det = float(np.linalg.det(self.Sigma))
        self._chol = None  # lazy

    @property
    def Sinv(self):
        return self._Sinv

    @property
    def det(self):
        return self._det

    @property
    def chol(self):
        if self._chol is None:
            self._chol = np.linalg.cholesky(self.Sigma)
        return self._chol

    def mahalanobis2(self, X):
        """Return M^2 for points X (...,2)."""
        X = np.asarray(X, float)
        dX = X - self.mu
        return np.einsum("...i,ij,...j->...", dX, self.Sinv, dX)

    def pdf(self, X):
        """Unnormalized-safe 2D Gaussian pdf at points X (...,2)."""
        M2 = self.mahalanobis2(X)
        const = 1.0 / (2.0 * np.pi * np.sqrt(self.det))
        return const * np.exp(-0.5 * M2)

    def whiten(self, X):
        """z = L^{-1}(x - mu) with LL^T = Sigma."""
        X = np.asarray(X, float)
        return np.linalg.solve(self.chol, (X - self.mu).T).T

    def ellipse(self, ax, p=0.95, **kw):
        """Add a confidence ellipse for mass p."""
        # Try SciPy, fallback to common quantiles if missing
        try:
            from scipy.stats import chi2  # type: ignore

            c = float(chi2.ppf(p, df=2))
        except Exception:
            lookup = {
                0.50: 1.386,
                0.68: 2.279,
                0.90: 4.605,
                0.95: 5.991,
                0.99: 9.210,
                0.997: 11.829,
            }
            c = lookup.get(p, 5.991)

        evals, evecs = np.linalg.eigh(self.Sigma)
        order = evals.argsort()[::-1]
        evals, evecs = evals[order], evecs[:, order]
        width, height = 2 * np.sqrt(c * evals)
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        e = Ellipse(
            xy=self.mu, width=width, height=height, angle=angle, fill=False, **kw
        )
        ax.add_patch(e)
        return e


class BivariateGrid:
    """Holds (U, V, Z) and provides statistical utilities with no repetition."""

    def __init__(self, U: np.ndarray, V: np.ndarray, Z: np.ndarray):
        self.U = np.asarray(U)
        self.V = np.asarray(V)
        self.Z = np.asarray(Z, float)
        if self.Z.sum() <= 0:
            raise ValueError("Z must have positive mass.")
        self.P = self.Z / self.Z.sum()
        self.UU, self.VV = np.meshgrid(self.U, self.V)  # shape = Z.shape
        self.X_grid = np.stack([self.UU, self.VV], axis=-1)  # (|V|,|U|,2)
        self.X = self.X_grid.reshape(-1, 2)  # (Ncells,2)
        self.w = self.P.reshape(-1)  # (Ncells,)

    # ---- moments / Gaussian fit ----
    def mean_cov(self):
        mu_u = (self.P * self.UU).sum()
        mu_v = (self.P * self.VV).sum()
        du, dv = self.UU - mu_u, self.VV - mu_v
        cov_uu = (self.P * du * du).sum()
        cov_vv = (self.P * dv * dv).sum()
        cov_uv = (self.P * du * dv).sum()
        mu = np.array([mu_u, mu_v])
        Sigma = np.array([[cov_uu, cov_uv], [cov_uv, cov_vv]])
        return mu, Sigma

    def gaussian_from_grid(self, return_eigendecomp=False):
        mu, Sigma = self.mean_cov()
        if not return_eigendecomp:
            return mu, Sigma
        evals, evecs = np.linalg.eigh(Sigma)
        order = evals.argsort()[::-1]
        return mu, Sigma, evals[order], evecs[:, order]

    # ---- discretize a Gaussian on this grid's support ----
    def discretize_gaussian(self, gauss: Gaussian2D, restrict_to_support=True):
        Q = gauss.pdf(self.X_grid)
        if restrict_to_support:
            Q = np.where(self.Z > 0, Q, 0.0)
        Q = Q / Q.sum()
        return Q

    # ---- metrics that involve Gaussian model ----
    def mahalanobis_ks(self, gauss: Gaussian2D):
        M2 = gauss.mahalanobis2(self.X)
        M2 = np.atleast_1d(M2)
        order = np.argsort(M2)
        m2_sorted = M2[order]
        w_sorted = self.w[order]
        cdf_hat = np.cumsum(w_sorted)
        F = 1.0 - np.exp(-0.5 * m2_sorted)  # chi2(df=2)
        KS = float(np.max(np.abs(cdf_hat - F)))
        return KS, m2_sorted, cdf_hat, F

    def mardia_moments(self, gauss: Gaussian2D):
        M2 = gauss.mahalanobis2(self.X)
        E_M2 = float(np.sum(self.w * M2))
        E_M4 = float(np.sum(self.w * (M2**2)))
        return E_M2, E_M4

    def angle_uniformity(self, gauss: Gaussian2D):
        Zw = gauss.whiten(self.X)
        theta = np.arctan2(Zw[:, 1], Zw[:, 0])
        c = float(np.sum(self.w * np.cos(theta)))
        s = float(np.sum(self.w * np.sin(theta)))
        R = np.sqrt(c * c + s * s)  # 0 is ideal
        return float(R)

    def mardia_skewness(self, gauss: Gaussian2D):
        Zw = gauss.whiten(self.X)
        G = Zw @ Zw.T  # pairwise inner products
        W = np.outer(self.w, self.w)
        beta1p = float(np.sum(W * (G**3)))
        return beta1p

    # ---- metrics that do not require a Gaussian ----
    def conditional_linearity(self):
        pV = self.P.sum(axis=1)  # shape |V|
        EU_cond = np.zeros_like(self.V, dtype=float)
        VarU_cond = np.zeros_like(self.V, dtype=float)
        for i in range(len(self.V)):
            if pV[i] > 0:
                pU_given_v = self.P[i, :] / pV[i]
                mu_u_v = (pU_given_v * self.U).sum()
                EU_cond[i] = mu_u_v
                VarU_cond[i] = (pU_given_v * (self.U - mu_u_v) ** 2).sum()

        w = pV
        X = np.vstack([np.ones_like(self.V, float), self.V]).T
        W = np.diag(w)
        beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ EU_cond)
        intercept, slope = float(beta[0]), float(beta[1])

        y = EU_cond
        yhat = X @ beta
        ybar = (w * y).sum() / w.sum()
        SS_res = float((w * (y - yhat) ** 2).sum())
        SS_tot = float((w * (y - ybar) ** 2).sum())
        R2 = 1.0 - SS_res / SS_tot if SS_tot > 0 else 1.0

        mean_var = float(np.average(VarU_cond, weights=w))
        std_var = float(np.sqrt(np.average((VarU_cond - mean_var) ** 2, weights=w)))
        cv_var = float(std_var / mean_var) if mean_var > 0 else 0.0

        return dict(
            intercept=intercept,
            slope=slope,
            R2=R2,
            mean_cond_var=mean_var,
            cv_cond_var=cv_var,
            EU_cond=EU_cond,
            VarU_cond=VarU_cond,
            weights=w,
        )

    # ---- builder for Z from AnalyticalResult (keeps original logic) ----
    @classmethod
    def from_analytical(cls, ar, N, sizes):
        sizes = list(sizes)
        m, S = len(sizes), sum(sizes)
        nmax, nmin = max(sizes), min(sizes)
        U = np.arange(nmax, min(N, S) + 1)
        V = np.arange(0, nmin + 1)
        den = prod(comb(N, n) for n in sizes)
        Z = np.zeros((len(V), len(U)), float)
        total_cases = 0

        for iv, v in enumerate(V):
            u_lo = max(nmax, (S - v + (m - 2)) // (m - 1))
            u_hi = min(N, S - (m - 1) * v)
            if u_lo > u_hi:
                continue
            for u in range(u_lo, u_hi + 1):
                cases = ar.bivariate_cases(u, v)
                total_cases += cases
                if cases:
                    Z[iv, u - U[0]] = cases / den

        grid = cls(U, V, Z)
        return grid, float(Z.sum()), (total_cases == den)


def compare_distributions(P, Q, eps=1e-300):
    P = P / P.sum()
    Q = Q / Q.sum()
    Qc = np.clip(Q, eps, None)
    TV = 0.5 * np.abs(P - Q).sum()
    maskP = P > 0
    KL_PQ = np.sum(P[maskP] * (np.log(P[maskP]) - np.log(Qc[maskP])))
    maskQ = Q > 0
    KL_QP = np.sum(Q[maskQ] * (np.log(Q[maskQ]) - np.log(np.clip(P[maskQ], eps, None))))
    M = 0.5 * (P + Q)
    JSD = 0.5 * (
        np.sum(P[maskP] * (np.log(P[maskP]) - np.log(np.clip(M[maskP], eps, None))))
        + np.sum(Q[maskQ] * (np.log(Q[maskQ]) - np.log(np.clip(M[maskQ], eps, None))))
    )
    CHI2 = np.sum((P - Q) ** 2 / np.clip(Q, eps, None))
    return dict(
        TV=float(TV),
        KL_PQ=float(KL_PQ),
        KL_QP=float(KL_QP),
        JSD=float(JSD),
        CHI2=float(CHI2),
    )
