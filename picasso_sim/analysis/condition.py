"""
Condition number analysis for photonic meshes.

The condition number κ(U) = σ_max / σ_min quantifies how small
perturbations are amplified. For unitary matrices, κ(U) = 1,
but the mesh Jacobian can have large condition numbers indicating
sensitivity to parameter changes.

This module analyzes:
1. Condition number of the mesh Jacobian
2. Singular value spectrum and distribution
3. Connections to random matrix theory (Marchenko-Pastur)

References:
- Ideas document: "Condition number statistics using tools from
  random matrix theory, such as the Marchenko-Pastur law"
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


def condition_number(A: np.ndarray) -> float:
    """
    Compute condition number of a matrix.

    κ(A) = σ_max / σ_min

    Parameters
    ----------
    A : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Condition number (≥ 1, or inf if singular).
    """
    try:
        return np.linalg.cond(A)
    except np.linalg.LinAlgError:
        return np.inf


def singular_value_spectrum(A: np.ndarray) -> np.ndarray:
    """
    Compute singular values of a matrix.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.

    Returns
    -------
    np.ndarray
        Singular values in descending order.
    """
    return np.linalg.svd(A, compute_uv=False)


@dataclass
class ConditionAnalysis:
    """
    Results of condition number analysis.

    Attributes
    ----------
    condition_number : float
        κ = σ_max / σ_min.
    singular_values : np.ndarray
        All singular values.
    sigma_max : float
        Largest singular value.
    sigma_min : float
        Smallest singular value.
    effective_rank : int
        Number of significant singular values.
    spectral_gap : float
        Gap between largest and second-largest singular values.
    """
    condition_number: float
    singular_values: np.ndarray
    sigma_max: float
    sigma_min: float
    effective_rank: int
    spectral_gap: float


def analyze_condition(A: np.ndarray, rank_threshold: float = 1e-10) -> ConditionAnalysis:
    """
    Comprehensive condition number analysis.

    Parameters
    ----------
    A : np.ndarray
        Input matrix.
    rank_threshold : float
        Threshold for determining effective rank.

    Returns
    -------
    ConditionAnalysis
        Analysis results.
    """
    svs = singular_value_spectrum(A)
    sigma_max = svs[0]
    sigma_min = svs[-1] if svs[-1] > rank_threshold else rank_threshold

    cond = sigma_max / sigma_min

    # Effective rank: number of singular values > threshold * sigma_max
    effective_rank = np.sum(svs > rank_threshold * sigma_max)

    # Spectral gap
    spectral_gap = svs[0] - svs[1] if len(svs) > 1 else svs[0]

    return ConditionAnalysis(
        condition_number=cond,
        singular_values=svs,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        effective_rank=effective_rank,
        spectral_gap=spectral_gap
    )


def mesh_jacobian_condition(mesh) -> Dict[str, float]:
    """
    Analyze condition number of the mesh Jacobian.

    The Jacobian J has shape (n_mzis, n*n) after flattening,
    or can be viewed as a tensor.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to analyze.

    Returns
    -------
    dict
        Condition analysis results.
    """
    from .sensitivity import compute_jacobian

    J_theta, J_phi = compute_jacobian(mesh, flatten=True)

    # Combined Jacobian
    J_combined = np.hstack([J_theta, J_phi])

    analysis = analyze_condition(J_combined)

    return {
        'jacobian_condition': analysis.condition_number,
        'jacobian_rank': analysis.effective_rank,
        'sigma_max': analysis.sigma_max,
        'sigma_min': analysis.sigma_min,
        'singular_values': analysis.singular_values
    }


def condition_vs_depth(mesh_class, n_modes: int,
                       depths: Optional[List[int]] = None,
                       n_samples: int = 50,
                       rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Analyze how Jacobian condition number scales with circuit depth.

    Parameters
    ----------
    mesh_class : class
        Mesh class to instantiate.
    n_modes : int
        Number of modes.
    depths : list of int, optional
        Circuit depths to test. Default uses mesh natural depth.
    n_samples : int
        Monte Carlo samples per depth.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        'depth': circuit depths
        'mean_condition': mean condition number
        'std_condition': standard deviation
    """
    if rng is None:
        rng = np.random.default_rng()

    mesh = mesh_class(n_modes)

    # Use natural depth if not specified
    if depths is None:
        depths = [mesh.depth]

    mean_conditions = []
    std_conditions = []

    for depth in depths:
        conditions = []

        for _ in range(n_samples):
            # Random phases
            thetas = rng.uniform(0, np.pi / 2, mesh.n_mzis)
            phis = rng.uniform(0, 2 * np.pi, mesh.n_mzis)
            mesh.set_phases(thetas, phis)

            result = mesh_jacobian_condition(mesh)
            conditions.append(result['jacobian_condition'])

        mean_conditions.append(np.mean(conditions))
        std_conditions.append(np.std(conditions))

    return {
        'depth': np.array(depths),
        'mean_condition': np.array(mean_conditions),
        'std_condition': np.array(std_conditions)
    }


def singular_value_distribution(mesh, n_samples: int = 100,
                                rng: Optional[np.random.Generator] = None
                                ) -> np.ndarray:
    """
    Sample the singular value distribution of the mesh Jacobian.

    For random MZI phases, the Jacobian becomes a random matrix
    and its singular values follow a distribution related to
    Marchenko-Pastur.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh.
    n_samples : int
        Number of random phase configurations.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        All sampled singular values (flattened).
    """
    if rng is None:
        rng = np.random.default_rng()

    all_svs = []

    for _ in range(n_samples):
        # Random phases
        thetas = rng.uniform(0, np.pi / 2, mesh.n_mzis)
        phis = rng.uniform(0, 2 * np.pi, mesh.n_mzis)
        mesh.set_phases(thetas, phis)

        result = mesh_jacobian_condition(mesh)
        all_svs.extend(result['singular_values'])

    return np.array(all_svs)


def perturbation_amplification(mesh, noise_model,
                               n_samples: int = 100,
                               rng: Optional[np.random.Generator] = None
                               ) -> Dict[str, np.ndarray]:
    """
    Empirically measure perturbation amplification.

    Compare ||δU|| / ||δθ, δφ|| across samples.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh.
    noise_model : NoiseModel
        Noise model to apply.
    n_samples : int
        Number of samples.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        'input_norms': ||δθ, δφ|| for each sample
        'output_norms': ||δU|| for each sample
        'amplification': ratio ||δU|| / ||δθ, δφ||
    """
    if rng is None:
        rng = np.random.default_rng()

    input_norms = []
    output_norms = []

    U_ideal = mesh.unitary(include_noise=False)

    for _ in range(n_samples):
        delta_thetas, delta_phis = noise_model.sample(mesh.n_mzis, rng)
        mesh.apply_noise(delta_thetas, delta_phis)

        U_noisy = mesh.unitary(include_noise=True)
        delta_U = U_noisy - U_ideal

        # Input norm
        input_norm = np.sqrt(np.sum(delta_thetas**2) + np.sum(delta_phis**2))
        input_norms.append(input_norm)

        # Output norm
        output_norm = np.linalg.norm(delta_U, 'fro')
        output_norms.append(output_norm)

        mesh.clear_noise()

    input_norms = np.array(input_norms)
    output_norms = np.array(output_norms)

    # Avoid division by zero
    amplification = np.divide(output_norms, input_norms,
                             out=np.zeros_like(output_norms),
                             where=input_norms > 1e-15)

    return {
        'input_norms': input_norms,
        'output_norms': output_norms,
        'amplification': amplification,
        'mean_amplification': np.mean(amplification),
        'max_amplification': np.max(amplification)
    }


def stability_margin(mesh, target_fidelity: float = 0.99,
                     noise_model_class=None,
                     rng: Optional[np.random.Generator] = None) -> float:
    """
    Estimate the noise tolerance for achieving target fidelity.

    Find σ such that E[F(U_noisy, U_ideal)] ≥ target_fidelity.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh.
    target_fidelity : float
        Desired minimum fidelity.
    noise_model_class : class, optional
        Noise model class with sigma_theta, sigma_phi attributes.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    float
        Maximum noise σ for which target fidelity is achieved.
    """
    if rng is None:
        rng = np.random.default_rng()

    if noise_model_class is None:
        try:
            from ..core.noise import GaussianPhaseNoise
        except ImportError:
            from picasso_sim.core.noise import GaussianPhaseNoise
        noise_model_class = GaussianPhaseNoise

    from .fidelity import fidelity

    U_ideal = mesh.unitary(include_noise=False)

    # Binary search for stability margin
    sigma_low, sigma_high = 0.0, 1.0
    n_samples = 50

    while sigma_high - sigma_low > 0.001:
        sigma_mid = (sigma_low + sigma_high) / 2
        noise = noise_model_class(sigma_theta=sigma_mid, sigma_phi=sigma_mid)

        fidelities = []
        for _ in range(n_samples):
            noise.apply_to_mesh(mesh, rng)
            U_noisy = mesh.unitary(include_noise=True)
            fidelities.append(fidelity(U_ideal, U_noisy))
            mesh.clear_noise()

        mean_fid = np.mean(fidelities)

        if mean_fid >= target_fidelity:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return sigma_low
