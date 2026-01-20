"""
Marchenko-Pastur distribution analysis.

The Marchenko-Pastur law describes the limiting distribution of
singular values for large random matrices. This is relevant for
analyzing the Jacobian of large photonic meshes.

For an M×N matrix X with i.i.d. entries, the eigenvalues of
(1/N)X†X converge to the MP distribution as M, N → ∞ with M/N → λ.

References:
- Marchenko & Pastur (1967), "Distribution of eigenvalues for some
  sets of random matrices"
- Ideas document: "Condition number statistics using tools from
  random matrix theory, such as the Marchenko-Pastur law"
"""

import numpy as np
from typing import Tuple, Optional
from scipy import stats, optimize


def marchenko_pastur_pdf(x: np.ndarray, lambda_ratio: float,
                         sigma: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.

    For λ = M/N (rows/cols ratio):
    - λ_± = σ²(1 ± √λ)²  (support bounds)
    - p(x) = √((λ₊-x)(x-λ₋)) / (2πλσ²x)  for x ∈ [λ₋, λ₊]

    Parameters
    ----------
    x : np.ndarray
        Points to evaluate density.
    lambda_ratio : float
        Aspect ratio M/N (should be ≤ 1 for standard form).
    sigma : float
        Variance parameter of matrix entries.

    Returns
    -------
    np.ndarray
        Probability density at each x.
    """
    if lambda_ratio > 1:
        lambda_ratio = 1 / lambda_ratio

    sigma2 = sigma ** 2

    # Support bounds
    lambda_minus = sigma2 * (1 - np.sqrt(lambda_ratio)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(lambda_ratio)) ** 2

    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lambda_minus) & (x <= lambda_plus) & (x > 0)

    if np.any(mask):
        x_valid = x[mask]
        pdf[mask] = (np.sqrt((lambda_plus - x_valid) * (x_valid - lambda_minus)) /
                     (2 * np.pi * lambda_ratio * sigma2 * x_valid))

    # Point mass at zero if λ < 1
    # (not included in continuous density)

    return pdf


def marchenko_pastur_cdf(x: np.ndarray, lambda_ratio: float,
                         sigma: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur cumulative distribution function.

    Computed via numerical integration of the PDF.

    Parameters
    ----------
    x : np.ndarray
        Points to evaluate CDF.
    lambda_ratio : float
        Aspect ratio M/N.
    sigma : float
        Variance parameter.

    Returns
    -------
    np.ndarray
        CDF values.
    """
    from scipy.integrate import quad

    if lambda_ratio > 1:
        lambda_ratio = 1 / lambda_ratio

    sigma2 = sigma ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(lambda_ratio)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(lambda_ratio)) ** 2

    cdf = np.zeros_like(x, dtype=float)

    # Point mass at zero
    point_mass = max(0, 1 - lambda_ratio)

    for i, xi in enumerate(x):
        if xi <= 0:
            cdf[i] = 0
        elif xi <= lambda_minus:
            cdf[i] = point_mass
        elif xi >= lambda_plus:
            cdf[i] = 1
        else:
            # Integrate PDF from lambda_minus to xi
            def integrand(t):
                return marchenko_pastur_pdf(np.array([t]), lambda_ratio, sigma)[0]
            integral, _ = quad(integrand, lambda_minus, xi)
            cdf[i] = point_mass + integral

    return cdf


def marchenko_pastur_bounds(lambda_ratio: float, sigma: float = 1.0
                           ) -> Tuple[float, float]:
    """
    Get support bounds of MP distribution.

    Parameters
    ----------
    lambda_ratio : float
        Aspect ratio M/N.
    sigma : float
        Variance parameter.

    Returns
    -------
    lambda_minus : float
        Lower bound of support.
    lambda_plus : float
        Upper bound of support.
    """
    if lambda_ratio > 1:
        lambda_ratio = 1 / lambda_ratio

    sigma2 = sigma ** 2
    lambda_minus = sigma2 * (1 - np.sqrt(lambda_ratio)) ** 2
    lambda_plus = sigma2 * (1 + np.sqrt(lambda_ratio)) ** 2

    return lambda_minus, lambda_plus


def fit_marchenko_pastur(eigenvalues: np.ndarray,
                         known_ratio: Optional[float] = None
                         ) -> dict:
    """
    Fit Marchenko-Pastur distribution to observed eigenvalues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Observed eigenvalues (squared singular values).
    known_ratio : float, optional
        If known, fix the aspect ratio.

    Returns
    -------
    dict
        'lambda_ratio': estimated aspect ratio
        'sigma': estimated variance parameter
        'ks_statistic': Kolmogorov-Smirnov statistic
        'ks_pvalue': KS test p-value
    """
    eigenvalues = np.asarray(eigenvalues)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove zeros

    # Estimate parameters
    mean_eig = np.mean(eigenvalues)
    var_eig = np.var(eigenvalues)

    if known_ratio is not None:
        lambda_ratio = known_ratio
        # Estimate sigma from mean: E[x] = σ²
        sigma = np.sqrt(mean_eig)
    else:
        # Method of moments estimation
        # E[x] = σ², Var[x] = σ⁴λ
        # Rough estimate
        sigma = np.sqrt(mean_eig)
        if sigma > 0 and mean_eig > 0:
            lambda_ratio = var_eig / (mean_eig ** 2)
            lambda_ratio = np.clip(lambda_ratio, 0.01, 1.0)
        else:
            lambda_ratio = 0.5

    # Refine with MLE or minimize KS statistic
    def neg_log_likelihood(params):
        lam, sig = params
        if lam <= 0 or lam > 2 or sig <= 0:
            return np.inf
        pdf_vals = marchenko_pastur_pdf(eigenvalues, lam, sig)
        pdf_vals = np.maximum(pdf_vals, 1e-15)
        return -np.sum(np.log(pdf_vals))

    try:
        result = optimize.minimize(neg_log_likelihood, [lambda_ratio, sigma],
                                   method='Nelder-Mead',
                                   options={'maxiter': 500})
        if result.success:
            lambda_ratio, sigma = result.x
    except:
        pass

    # KS test
    cdf_theoretical = marchenko_pastur_cdf(np.sort(eigenvalues), lambda_ratio, sigma)
    n = len(eigenvalues)
    cdf_empirical = np.arange(1, n + 1) / n
    ks_stat = np.max(np.abs(cdf_theoretical - cdf_empirical))

    # Approximate p-value (for large n)
    ks_pvalue = 2 * np.exp(-2 * n * ks_stat ** 2)

    return {
        'lambda_ratio': lambda_ratio,
        'sigma': sigma,
        'ks_statistic': ks_stat,
        'ks_pvalue': ks_pvalue,
        'bounds': marchenko_pastur_bounds(lambda_ratio, sigma)
    }


def mp_condition_number_expected(lambda_ratio: float) -> float:
    """
    Expected condition number from MP distribution.

    κ = √(λ₊/λ₋) = (1 + √λ)/(1 - √λ)  for λ ≤ 1

    Parameters
    ----------
    lambda_ratio : float
        Aspect ratio.

    Returns
    -------
    float
        Expected condition number.
    """
    if lambda_ratio > 1:
        lambda_ratio = 1 / lambda_ratio

    if lambda_ratio >= 1:
        return np.inf

    sqrt_lambda = np.sqrt(lambda_ratio)
    return (1 + sqrt_lambda) / (1 - sqrt_lambda)


def compare_to_mp(mesh, n_samples: int = 100,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """
    Compare mesh Jacobian singular values to Marchenko-Pastur.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to analyze.
    n_samples : int
        Number of random configurations.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        Comparison statistics.
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        from ..analysis.condition import singular_value_distribution
    except ImportError:
        from picasso_sim.analysis.condition import singular_value_distribution

    # Collect singular values
    all_svs = singular_value_distribution(mesh, n_samples, rng)

    # Convert to eigenvalues (squared singular values)
    eigenvalues = all_svs ** 2

    # Fit MP
    fit_result = fit_marchenko_pastur(eigenvalues)

    # Compute histograms for comparison
    lambda_minus, lambda_plus = fit_result['bounds']
    x_theory = np.linspace(max(0.001, lambda_minus * 0.9),
                           lambda_plus * 1.1, 200)
    pdf_theory = marchenko_pastur_pdf(x_theory, fit_result['lambda_ratio'],
                                      fit_result['sigma'])

    return {
        'eigenvalues': eigenvalues,
        'fit_result': fit_result,
        'x_theory': x_theory,
        'pdf_theory': pdf_theory,
        'expected_condition': mp_condition_number_expected(fit_result['lambda_ratio'])
    }


def tracy_widom_edge(n: int, beta: int = 2) -> Tuple[float, float]:
    """
    Tracy-Widom scaling at the edge of MP distribution.

    The largest eigenvalue fluctuates as:
    λ_max ≈ λ₊ + n^(-2/3) · σ · TW_β

    Parameters
    ----------
    n : int
        Matrix dimension.
    beta : int
        Dyson index (1, 2, or 4).

    Returns
    -------
    scale : float
        Fluctuation scale n^(-2/3).
    location : float
        Edge location (for σ=1, λ=1).
    """
    scale = n ** (-2/3)
    location = 4.0  # (1 + 1)² for λ=1

    return scale, location


def eigenvalue_outlier_detection(eigenvalues: np.ndarray,
                                 lambda_ratio: float,
                                 sigma: float = 1.0,
                                 threshold: float = 3.0) -> np.ndarray:
    """
    Detect eigenvalues that are outliers from MP bulk.

    Outliers indicate localized errors or structural issues.

    Parameters
    ----------
    eigenvalues : np.ndarray
        Observed eigenvalues.
    lambda_ratio : float
        Aspect ratio.
    sigma : float
        Variance parameter.
    threshold : float
        Number of standard deviations beyond MP bounds.

    Returns
    -------
    np.ndarray
        Boolean mask of outliers.
    """
    lambda_minus, lambda_plus = marchenko_pastur_bounds(lambda_ratio, sigma)

    # Tracy-Widom scale for edge fluctuations
    n = len(eigenvalues)
    tw_scale = sigma * n ** (-2/3)

    lower_bound = lambda_minus - threshold * tw_scale
    upper_bound = lambda_plus + threshold * tw_scale

    outliers = (eigenvalues < lower_bound) | (eigenvalues > upper_bound)

    return outliers
