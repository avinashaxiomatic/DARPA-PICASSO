"""
Robust Large-Scale Bayesian Calibration for Photonic Meshes

This module implements numerically stable Bayesian inference for
error localization and correction in large MZI meshes (1000+ elements).

Key algorithms:
1. Tikhonov-regularized least squares with optimal λ selection
2. Conjugate gradient solver for large systems
3. Iterative refinement with convergence monitoring
4. Ensemble Kalman filter for uncertainty quantification
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy import linalg
from scipy.sparse.linalg import cg, lsqr
from scipy.optimize import minimize_scalar


@dataclass
class CalibrationResult:
    """Results from Bayesian calibration."""
    estimated_errors: np.ndarray  # Estimated phase errors
    uncertainties: np.ndarray     # Uncertainty in estimates
    residual_norm: float          # ||y - J·θ||
    iterations: int               # Number of iterations used
    converged: bool               # Whether solver converged
    fidelity_before: float        # Fidelity before correction
    fidelity_after: float         # Fidelity after correction


class RobustBayesianCalibrator:
    """
    Robust Bayesian calibrator for large-scale photonic meshes.

    Uses Tikhonov regularization with automatic parameter selection
    and iterative refinement for numerical stability.
    """

    def __init__(self, n_mzis: int, sigma_prior: float = 0.05):
        """
        Initialize calibrator.

        Parameters
        ----------
        n_mzis : int
            Number of MZIs in the mesh.
        sigma_prior : float
            Prior standard deviation on phase errors (radians).
        """
        self.n_mzis = n_mzis
        self.sigma_prior = sigma_prior

        # Accumulated measurements
        self.measurements = []  # List of (J, delta_U) pairs
        self.estimates = np.zeros(n_mzis)
        self.covariance = np.eye(n_mzis) * sigma_prior**2

    def add_measurement(self, jacobian: np.ndarray, delta_U: np.ndarray):
        """
        Add a measurement for batch processing.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix, shape (n_obs, n_mzis).
        delta_U : np.ndarray
            Observed unitary deviation, flattened.
        """
        self.measurements.append((jacobian.real, delta_U.real.flatten()))

    def solve(self, method: str = 'tikhonov',
              max_iter: int = 100,
              tol: float = 1e-6) -> np.ndarray:
        """
        Solve for error estimates using accumulated measurements.

        Parameters
        ----------
        method : str
            Solver method: 'tikhonov', 'conjugate_gradient', or 'iterative'.
        max_iter : int
            Maximum iterations for iterative methods.
        tol : float
            Convergence tolerance.

        Returns
        -------
        np.ndarray
            Estimated phase errors.
        """
        if not self.measurements:
            return self.estimates

        # Stack all measurements
        J_list = [m[0] for m in self.measurements]
        y_list = [m[1] for m in self.measurements]

        J = np.vstack(J_list)
        y = np.concatenate(y_list)

        if method == 'tikhonov':
            self.estimates = self._solve_tikhonov(J, y)
        elif method == 'conjugate_gradient':
            self.estimates = self._solve_cg(J, y, max_iter, tol)
        elif method == 'iterative':
            self.estimates = self._solve_iterative(J, y, max_iter, tol)
        else:
            raise ValueError(f"Unknown method: {method}")

        return self.estimates

    def _solve_tikhonov(self, J: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve using Tikhonov regularization with optimal λ.

        min ||y - J·θ||² + λ||θ||²

        Solution: θ = (J^T J + λI)^{-1} J^T y
        """
        # Find optimal regularization parameter using GCV
        lambda_opt = self._find_optimal_lambda(J, y)

        # Solve regularized system
        JtJ = J.T @ J
        Jty = J.T @ y

        # Add regularization
        A = JtJ + lambda_opt * np.eye(self.n_mzis)

        try:
            theta = linalg.solve(A, Jty, assume_a='pos')
        except linalg.LinAlgError:
            # Fall back to least squares
            theta = linalg.lstsq(J, y, lapack_driver='gelsy')[0]

        return theta

    def _find_optimal_lambda(self, J: np.ndarray, y: np.ndarray,
                             n_lambdas: int = 50) -> float:
        """
        Find optimal regularization parameter using Generalized Cross-Validation.

        GCV score: GCV(λ) = ||y - J·θ_λ||² / (1 - tr(H_λ)/n)²
        where H_λ = J(J^T J + λI)^{-1}J^T is the hat matrix.
        """
        # SVD of J for efficient computation
        try:
            U, s, Vt = linalg.svd(J, full_matrices=False)
        except linalg.LinAlgError:
            # If SVD fails, use default lambda
            return self.n_mzis * 0.01

        # Transform y to SVD basis
        y_transformed = U.T @ y

        n = len(y)

        def gcv_score(log_lambda):
            lam = np.exp(log_lambda)

            # Filter factors
            f = s**2 / (s**2 + lam)

            # Residual norm squared
            residual = y_transformed * (1 - f)
            rss = np.sum(residual**2) + np.sum(y**2) - np.sum(y_transformed**2)

            # Effective degrees of freedom
            df = np.sum(f)

            # GCV score
            denom = (1 - df/n)**2
            if denom < 1e-10:
                return np.inf

            return rss / (n * denom)

        # Search over log-scale
        log_lambdas = np.linspace(-10, 5, n_lambdas)
        scores = [gcv_score(ll) for ll in log_lambdas]

        best_idx = np.argmin(scores)
        lambda_opt = np.exp(log_lambdas[best_idx])

        # Bound lambda to reasonable range
        lambda_opt = np.clip(lambda_opt, 1e-8, 1e3)

        return lambda_opt

    def _solve_cg(self, J: np.ndarray, y: np.ndarray,
                  max_iter: int, tol: float) -> np.ndarray:
        """
        Solve using conjugate gradient on normal equations.

        Solves: (J^T J + λI) θ = J^T y
        """
        lambda_reg = self.sigma_prior**(-2)  # Prior precision as regularization

        JtJ = J.T @ J
        Jty = J.T @ y

        # Regularized matrix
        A = JtJ + lambda_reg * np.eye(self.n_mzis)

        # Solve with CG
        theta, info = cg(A, Jty, x0=self.estimates, maxiter=max_iter, rtol=tol)

        return theta

    def _solve_iterative(self, J: np.ndarray, y: np.ndarray,
                         max_iter: int, tol: float) -> np.ndarray:
        """
        Iterative refinement with adaptive damping.
        """
        theta = self.estimates.copy()

        # Compute optimal lambda once
        lambda_reg = self._find_optimal_lambda(J, y)

        JtJ = J.T @ J
        Jty = J.T @ y
        A = JtJ + lambda_reg * np.eye(self.n_mzis)

        # Precompute Cholesky factorization
        try:
            L = linalg.cholesky(A, lower=True)
            use_cholesky = True
        except linalg.LinAlgError:
            use_cholesky = False

        prev_residual = np.inf

        for iteration in range(max_iter):
            # Current residual
            residual = y - J @ theta
            residual_norm = np.linalg.norm(residual)

            # Check convergence
            if residual_norm < tol:
                break

            # Adaptive damping based on residual improvement
            if residual_norm < prev_residual:
                damping = 0.8
            else:
                damping = 0.3

            prev_residual = residual_norm

            # Solve for update
            rhs = Jty - JtJ @ theta

            if use_cholesky:
                delta = linalg.cho_solve((L, True), rhs)
            else:
                delta = linalg.solve(A, rhs)

            # Damped update
            theta = theta + damping * delta

            # Project to prior bounds
            max_error = 5 * self.sigma_prior
            theta = np.clip(theta, -max_error, max_error)

        return theta

    def get_uncertainties(self, J: np.ndarray) -> np.ndarray:
        """
        Compute posterior uncertainties on estimates.

        Returns standard deviation for each MZI error estimate.
        """
        lambda_reg = self.sigma_prior**(-2)

        JtJ = J.T @ J
        A = JtJ + lambda_reg * np.eye(self.n_mzis)

        try:
            A_inv = linalg.inv(A)
            variances = np.diag(A_inv)
            return np.sqrt(np.maximum(variances, 0))
        except linalg.LinAlgError:
            return np.ones(self.n_mzis) * self.sigma_prior

    def reset(self):
        """Reset accumulated measurements."""
        self.measurements = []
        self.estimates = np.zeros(self.n_mzis)


class EnsembleKalmanCalibrator:
    """
    Ensemble Kalman Filter for Bayesian calibration.

    Uses an ensemble of particles to represent the posterior distribution,
    providing both estimates and uncertainty quantification.

    More robust for nonlinear systems and large state spaces.
    """

    def __init__(self, n_mzis: int, sigma_prior: float = 0.05,
                 n_ensemble: int = 100):
        """
        Initialize ensemble Kalman filter.

        Parameters
        ----------
        n_mzis : int
            Number of MZIs.
        sigma_prior : float
            Prior standard deviation.
        n_ensemble : int
            Number of ensemble members.
        """
        self.n_mzis = n_mzis
        self.sigma_prior = sigma_prior
        self.n_ensemble = n_ensemble

        # Initialize ensemble from prior
        self.rng = np.random.default_rng(42)
        self.ensemble = self.rng.normal(0, sigma_prior, (n_ensemble, n_mzis))

    def update(self, jacobian: np.ndarray, observation: np.ndarray,
               obs_noise: float = 0.01):
        """
        Ensemble Kalman update.

        Parameters
        ----------
        jacobian : np.ndarray
            Observation operator (Jacobian), shape (n_obs, n_mzis).
        observation : np.ndarray
            Observed deviation.
        obs_noise : float
            Observation noise standard deviation.
        """
        J = jacobian.real
        y = observation.real.flatten()
        n_obs = len(y)

        # Predicted observations for each ensemble member
        # H(x_i) = J @ x_i
        predicted = self.ensemble @ J.T  # Shape: (n_ensemble, n_obs)

        # Ensemble mean and anomalies
        x_mean = np.mean(self.ensemble, axis=0)
        X_anomaly = self.ensemble - x_mean  # (n_ensemble, n_mzis)

        pred_mean = np.mean(predicted, axis=0)
        Y_anomaly = predicted - pred_mean  # (n_ensemble, n_obs)

        # Sample covariances
        # P_xy = X' Y'^T / (n-1)
        # P_yy = Y' Y'^T / (n-1) + R

        Pxy = X_anomaly.T @ Y_anomaly / (self.n_ensemble - 1)  # (n_mzis, n_obs)
        Pyy = Y_anomaly.T @ Y_anomaly / (self.n_ensemble - 1)  # (n_obs, n_obs)

        # Add observation noise
        R = obs_noise**2 * np.eye(n_obs)
        Pyy_reg = Pyy + R

        # Kalman gain: K = P_xy @ P_yy^{-1}
        try:
            K = Pxy @ linalg.solve(Pyy_reg, np.eye(n_obs), assume_a='pos')
        except linalg.LinAlgError:
            K = Pxy @ linalg.pinv(Pyy_reg)

        # Update each ensemble member with perturbed observations
        for i in range(self.n_ensemble):
            # Perturb observation
            y_perturbed = y + self.rng.normal(0, obs_noise, n_obs)

            # Innovation
            innovation = y_perturbed - predicted[i]

            # Update
            self.ensemble[i] += K @ innovation

        # Clip to reasonable range
        max_error = 5 * self.sigma_prior
        self.ensemble = np.clip(self.ensemble, -max_error, max_error)

    def get_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get posterior mean and standard deviation.

        Returns
        -------
        mean : np.ndarray
            Posterior mean estimate.
        std : np.ndarray
            Posterior standard deviation.
        """
        mean = np.mean(self.ensemble, axis=0)
        std = np.std(self.ensemble, axis=0)
        return mean, std

    def get_confidence_intervals(self, alpha: float = 0.05
                                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get confidence intervals from ensemble.

        Parameters
        ----------
        alpha : float
            Significance level (default 0.05 for 95% CI).

        Returns
        -------
        lower : np.ndarray
            Lower bound of CI.
        upper : np.ndarray
            Upper bound of CI.
        """
        lower = np.percentile(self.ensemble, 100 * alpha/2, axis=0)
        upper = np.percentile(self.ensemble, 100 * (1 - alpha/2), axis=0)
        return lower, upper

    def reset(self):
        """Reset ensemble to prior."""
        self.ensemble = self.rng.normal(0, self.sigma_prior,
                                        (self.n_ensemble, self.n_mzis))


class HybridCalibrator:
    """
    Hybrid calibrator combining multiple methods for robustness.

    1. First pass: Tikhonov for quick initial estimate
    2. Refinement: Ensemble Kalman for uncertainty
    3. Final: Iterative refinement for precision
    """

    def __init__(self, n_mzis: int, sigma_prior: float = 0.05):
        self.n_mzis = n_mzis
        self.sigma_prior = sigma_prior

        self.tikhonov = RobustBayesianCalibrator(n_mzis, sigma_prior)
        self.enkf = EnsembleKalmanCalibrator(n_mzis, sigma_prior, n_ensemble=50)

        self.estimates = np.zeros(n_mzis)
        self.uncertainties = np.ones(n_mzis) * sigma_prior

    def calibrate(self, jacobian: np.ndarray, observations: List[np.ndarray],
                  n_iterations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full calibration pipeline.

        Parameters
        ----------
        jacobian : np.ndarray
            Jacobian matrix.
        observations : list of np.ndarray
            Multiple observations of delta_U.
        n_iterations : int
            Number of EnKF iterations.

        Returns
        -------
        estimates : np.ndarray
            Error estimates.
        uncertainties : np.ndarray
            Uncertainty estimates.
        """
        J = jacobian.real

        # Step 1: Tikhonov for initial estimate
        for obs in observations:
            self.tikhonov.add_measurement(J, obs)

        initial_estimate = self.tikhonov.solve(method='tikhonov')

        # Step 2: EnKF refinement
        # Initialize ensemble around Tikhonov estimate
        self.enkf.ensemble = initial_estimate + self.enkf.rng.normal(
            0, self.sigma_prior * 0.5, (self.enkf.n_ensemble, self.n_mzis)
        )

        for i, obs in enumerate(observations[:n_iterations]):
            self.enkf.update(J, obs, obs_noise=0.01)

        self.estimates, self.uncertainties = self.enkf.get_estimates()

        return self.estimates, self.uncertainties

    def reset(self):
        self.tikhonov.reset()
        self.enkf.reset()
        self.estimates = np.zeros(self.n_mzis)
        self.uncertainties = np.ones(self.n_mzis) * self.sigma_prior


def calibrate_mesh(mesh, true_errors: np.ndarray,
                   n_measurements: int = 10,
                   method: str = 'hybrid',
                   sigma_prior: float = None) -> CalibrationResult:
    """
    High-level calibration function for a photonic mesh.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to calibrate.
    true_errors : np.ndarray
        True phase errors (for simulation).
    n_measurements : int
        Number of measurements to take.
    method : str
        Calibration method: 'tikhonov', 'enkf', or 'hybrid'.
    sigma_prior : float, optional
        Prior on error magnitude. If None, estimated from true_errors.

    Returns
    -------
    CalibrationResult
        Calibration results including estimates and fidelity improvement.
    """
    from .sensitivity import compute_jacobian
    from .fidelity import fidelity

    n_mzis = mesh.n_mzis

    if sigma_prior is None:
        sigma_prior = np.std(true_errors) * 2

    # Store original phases
    original_thetas = np.array([mzi.theta for mzi in mesh.mzis])
    original_phis = np.array([mzi.phi for mzi in mesh.mzis])

    # Ideal unitary
    U_ideal = mesh.unitary(include_noise=False)

    # Fidelity before correction
    mesh.apply_noise(true_errors, np.zeros(n_mzis))
    U_noisy = mesh.unitary(include_noise=True)
    fid_before = fidelity(U_ideal, U_noisy)
    mesh.clear_noise()

    # Initialize calibrator
    if method == 'tikhonov':
        calibrator = RobustBayesianCalibrator(n_mzis, sigma_prior)
    elif method == 'enkf':
        calibrator = EnsembleKalmanCalibrator(n_mzis, sigma_prior)
    elif method == 'hybrid':
        calibrator = HybridCalibrator(n_mzis, sigma_prior)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Collect measurements
    rng = np.random.default_rng()
    observations = []

    for m in range(n_measurements):
        # Compute Jacobian
        J_theta, J_phi = compute_jacobian(mesh, flatten=True)
        J = J_theta.T  # Shape: (n², n_mzis)

        # Apply noise and observe
        mesh.apply_noise(true_errors, np.zeros(n_mzis))
        U_noisy = mesh.unitary(include_noise=True)
        delta_U = (U_noisy - U_ideal).flatten()
        observations.append(delta_U)
        mesh.clear_noise()

        # Perturb phases slightly for next measurement
        perturb = rng.normal(0, 0.02, n_mzis)
        mesh.set_phases(original_thetas + perturb, original_phis)
        U_ideal = mesh.unitary(include_noise=False)

    # Solve
    if method == 'hybrid':
        estimates, uncertainties = calibrator.calibrate(J, observations)
    elif method == 'enkf':
        for obs in observations:
            calibrator.update(J, obs)
        estimates, uncertainties = calibrator.get_estimates()
    else:
        for obs in observations:
            calibrator.add_measurement(J, obs)
        estimates = calibrator.solve()
        uncertainties = calibrator.get_uncertainties(J)

    # Apply correction
    mesh.set_phases(original_thetas, original_phis)
    U_ideal = mesh.unitary(include_noise=False)

    correction = -estimates * 0.8  # Slight damping
    mesh.apply_noise(true_errors + correction, np.zeros(n_mzis))
    U_corrected = mesh.unitary(include_noise=True)
    fid_after = fidelity(U_ideal, U_corrected)
    mesh.clear_noise()

    # Compute residual
    residual = true_errors - estimates
    residual_norm = np.linalg.norm(residual) / np.linalg.norm(true_errors)

    return CalibrationResult(
        estimated_errors=estimates,
        uncertainties=uncertainties,
        residual_norm=residual_norm,
        iterations=n_measurements,
        converged=residual_norm < 0.5,
        fidelity_before=fid_before,
        fidelity_after=fid_after
    )
