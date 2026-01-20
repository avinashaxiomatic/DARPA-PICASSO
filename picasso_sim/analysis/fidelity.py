"""
Fidelity metrics for comparing unitary transformations.

This module provides various metrics for quantifying how close
a perturbed mesh unitary is to a target unitary.

Key metrics from the ideas document:
1. Process fidelity: F = |Tr(U†_mesh U_target)|² / N²
2. Operator distance: ||U_mesh - U_target||
3. Diamond norm distance (for quantum channels)
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


def fidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute the process fidelity between two unitaries.

    F(U1, U2) = |Tr(U1† U2)|² / N²

    This is the probability that U2 produces the same output as U1
    averaged over all pure states (Haar measure).

    Parameters
    ----------
    U1 : np.ndarray
        First unitary matrix.
    U2 : np.ndarray
        Second unitary matrix.

    Returns
    -------
    float
        Fidelity in [0, 1]. F=1 means U1=U2 (up to global phase).

    Examples
    --------
    >>> I = np.eye(4)
    >>> fidelity(I, I)
    1.0
    """
    n = U1.shape[0]
    overlap = np.trace(U1.conj().T @ U2)
    return np.abs(overlap) ** 2 / (n ** 2)


def process_fidelity(U_target: np.ndarray, U_actual: np.ndarray) -> float:
    """
    Compute process fidelity (same as fidelity but named for clarity).

    Parameters
    ----------
    U_target : np.ndarray
        Target unitary.
    U_actual : np.ndarray
        Actual (possibly perturbed) unitary.

    Returns
    -------
    float
        Process fidelity.
    """
    return fidelity(U_target, U_actual)


def operator_distance(U1: np.ndarray, U2: np.ndarray,
                     norm: str = 'fro') -> float:
    """
    Compute distance between two unitaries.

    Parameters
    ----------
    U1, U2 : np.ndarray
        Unitary matrices.
    norm : str
        Norm type: 'fro' (Frobenius), '2' (spectral), 'nuc' (nuclear).

    Returns
    -------
    float
        Distance ||U1 - U2||.
    """
    diff = U1 - U2

    if norm == 'fro':
        return np.linalg.norm(diff, 'fro')
    elif norm == '2' or norm == 'spectral':
        return np.linalg.norm(diff, 2)
    elif norm == 'nuc' or norm == 'nuclear':
        return np.linalg.norm(diff, 'nuc')
    else:
        raise ValueError(f"Unknown norm: {norm}")


def trace_distance(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute trace distance between unitaries.

    D(U1, U2) = (1/2) ||U1 - U2||_1 = (1/2) Tr(|U1 - U2|)

    Parameters
    ----------
    U1, U2 : np.ndarray
        Unitary matrices.

    Returns
    -------
    float
        Trace distance.
    """
    diff = U1 - U2
    # |A| = sqrt(A† A)
    singular_values = np.linalg.svd(diff, compute_uv=False)
    return 0.5 * np.sum(singular_values)


def average_gate_fidelity(U_target: np.ndarray, U_actual: np.ndarray) -> float:
    """
    Compute average gate fidelity.

    F_avg = (d·F + 1) / (d + 1)

    where d is the dimension and F is the process fidelity.
    This accounts for the identity component.

    Parameters
    ----------
    U_target : np.ndarray
        Target unitary.
    U_actual : np.ndarray
        Actual unitary.

    Returns
    -------
    float
        Average gate fidelity.
    """
    d = U_target.shape[0]
    F = process_fidelity(U_target, U_actual)
    return (d * F + 1) / (d + 1)


def infidelity(U1: np.ndarray, U2: np.ndarray) -> float:
    """
    Compute infidelity: 1 - F.

    Parameters
    ----------
    U1, U2 : np.ndarray
        Unitary matrices.

    Returns
    -------
    float
        Infidelity in [0, 1].
    """
    return 1 - fidelity(U1, U2)


def state_fidelity(psi1: np.ndarray, psi2: np.ndarray) -> float:
    """
    Compute fidelity between two pure states.

    F = |⟨ψ1|ψ2⟩|²

    Parameters
    ----------
    psi1, psi2 : np.ndarray
        State vectors (normalized).

    Returns
    -------
    float
        State fidelity.
    """
    return np.abs(np.vdot(psi1, psi2)) ** 2


def worst_case_fidelity(U_target: np.ndarray, U_actual: np.ndarray,
                        n_samples: int = 1000,
                        rng: Optional[np.random.Generator] = None) -> Tuple[float, np.ndarray]:
    """
    Estimate worst-case fidelity over all input states.

    F_min = min_|ψ⟩ |⟨ψ|U†_target U_actual|ψ⟩|²

    Parameters
    ----------
    U_target : np.ndarray
        Target unitary.
    U_actual : np.ndarray
        Actual unitary.
    n_samples : int
        Number of random states to sample.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    min_fidelity : float
        Estimated worst-case fidelity.
    worst_state : np.ndarray
        State achieving worst fidelity.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = U_target.shape[0]
    W = U_target.conj().T @ U_actual  # U†_target U_actual

    min_fid = 1.0
    worst_psi = None

    for _ in range(n_samples):
        # Random state from Haar measure
        psi = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        psi /= np.linalg.norm(psi)

        fid = np.abs(np.vdot(psi, W @ psi)) ** 2
        if fid < min_fid:
            min_fid = fid
            worst_psi = psi.copy()

    return min_fid, worst_psi


@dataclass
class FidelityReport:
    """
    Comprehensive fidelity report for a mesh.

    Attributes
    ----------
    process_fidelity : float
        Standard process fidelity.
    average_gate_fidelity : float
        Average gate fidelity.
    worst_case_fidelity : float
        Estimated worst-case fidelity.
    trace_distance : float
        Trace distance.
    frobenius_distance : float
        Frobenius norm distance.
    spectral_distance : float
        Spectral (operator) norm distance.
    """
    process_fidelity: float
    average_gate_fidelity: float
    worst_case_fidelity: float
    trace_distance: float
    frobenius_distance: float
    spectral_distance: float

    def __repr__(self) -> str:
        return (f"FidelityReport(\n"
                f"  process_fidelity={self.process_fidelity:.6f},\n"
                f"  average_gate_fidelity={self.average_gate_fidelity:.6f},\n"
                f"  worst_case_fidelity={self.worst_case_fidelity:.6f},\n"
                f"  trace_distance={self.trace_distance:.6f},\n"
                f"  frobenius_distance={self.frobenius_distance:.6f},\n"
                f"  spectral_distance={self.spectral_distance:.6f}\n)")


def comprehensive_fidelity_report(U_target: np.ndarray, U_actual: np.ndarray,
                                  n_samples: int = 1000,
                                  rng: Optional[np.random.Generator] = None
                                  ) -> FidelityReport:
    """
    Generate comprehensive fidelity report.

    Parameters
    ----------
    U_target : np.ndarray
        Target unitary.
    U_actual : np.ndarray
        Actual unitary.
    n_samples : int
        Samples for worst-case estimation.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    FidelityReport
        Comprehensive report.
    """
    proc_fid = process_fidelity(U_target, U_actual)
    avg_fid = average_gate_fidelity(U_target, U_actual)
    worst_fid, _ = worst_case_fidelity(U_target, U_actual, n_samples, rng)

    return FidelityReport(
        process_fidelity=proc_fid,
        average_gate_fidelity=avg_fid,
        worst_case_fidelity=worst_fid,
        trace_distance=trace_distance(U_target, U_actual),
        frobenius_distance=operator_distance(U_target, U_actual, 'fro'),
        spectral_distance=operator_distance(U_target, U_actual, '2')
    )


def fidelity_vs_noise(mesh, noise_model, noise_scales: np.ndarray,
                      n_samples: int = 100,
                      rng: Optional[np.random.Generator] = None) -> dict:
    """
    Compute fidelity as a function of noise strength.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to analyze.
    noise_model : NoiseModel
        Noise model (will be scaled).
    noise_scales : np.ndarray
        Array of noise scaling factors.
    n_samples : int
        Monte Carlo samples per scale.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        'scales': noise scales
        'mean_fidelity': mean fidelity at each scale
        'std_fidelity': standard deviation
    """
    if rng is None:
        rng = np.random.default_rng()

    U_ideal = mesh.unitary(include_noise=False)

    mean_fidelities = []
    std_fidelities = []

    # Store original noise parameters if applicable
    original_sigma_theta = getattr(noise_model, 'sigma_theta', 0.01)
    original_sigma_phi = getattr(noise_model, 'sigma_phi', 0.01)

    for scale in noise_scales:
        fidelities = []

        # Scale noise
        if hasattr(noise_model, 'sigma_theta'):
            noise_model.sigma_theta = original_sigma_theta * scale
        if hasattr(noise_model, 'sigma_phi'):
            noise_model.sigma_phi = original_sigma_phi * scale

        for _ in range(n_samples):
            noise_model.apply_to_mesh(mesh, rng)
            U_noisy = mesh.unitary(include_noise=True)
            fidelities.append(fidelity(U_ideal, U_noisy))
            mesh.clear_noise()

        mean_fidelities.append(np.mean(fidelities))
        std_fidelities.append(np.std(fidelities))

    # Restore original parameters
    if hasattr(noise_model, 'sigma_theta'):
        noise_model.sigma_theta = original_sigma_theta
    if hasattr(noise_model, 'sigma_phi'):
        noise_model.sigma_phi = original_sigma_phi

    return {
        'scales': noise_scales,
        'mean_fidelity': np.array(mean_fidelities),
        'std_fidelity': np.array(std_fidelities)
    }
