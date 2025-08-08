import numpy as np
import scipy.stats as stats
from scipy.stats import norm


def mu_calculation(x_values: list[float], pmf: list[float]) -> float:
    x_arr = np.array(x_values, dtype=float)
    p_arr = np.array(pmf, dtype=float)
    return float(np.sum(x_arr * p_arr))


def sigma_calculation(x_values: list[float], pmf: list[float]) -> float:
    mu = mu_calculation(x_values, pmf)
    x_arr = np.array(x_values, dtype=float)
    p_arr = np.array(pmf, dtype=float)
    return float(np.sqrt(np.sum(((x_arr - mu) ** 2) * p_arr)))


def mae_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    normal_pdf_arr = np.array(normal_pdf, dtype=float)
    approx_pdf_arr = np.array(approx_pdf, dtype=float)
    return np.mean(np.abs(normal_pdf_arr - approx_pdf_arr))  # Mean absolute error


def sse_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    normal_pdf_arr = np.array(normal_pdf, dtype=float)
    approx_pdf_arr = np.array(approx_pdf, dtype=float)
    return np.sum((normal_pdf_arr - approx_pdf_arr) ** 2)  # Sum of squared errors


def norm_pdf(x: float, mu: float, sigma: float) -> float:
    if sigma is None or sigma <= 0:
        return np.zeros_like(x, dtype=float)
    return stats.norm.pdf(x, mu, sigma)


def discretize_normal_pmf(x_vals: list[float], mu: float, sigma: float) -> list[float]:
    x = np.asarray(sorted(x_vals), dtype=float)
    if sigma is None or sigma <= 0:
        return np.zeros_like(x)

    # Calculatecell boundaries (midpoints)
    mids = (x[1:] + x[:-1]) / 2
    edges = np.concatenate(([-np.inf], mids, [np.inf]))

    # Calculatemass per cell = CDF difference
    return np.diff(norm.cdf(edges, loc=mu, scale=sigma))
