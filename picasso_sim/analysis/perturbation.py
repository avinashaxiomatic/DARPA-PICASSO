"""
Perturbation analysis for photonic MZI meshes.

This module implements first-order perturbation theory to analyze how
local phase errors propagate through the mesh to affect the global
unitary transformation.

Key mathematical framework:
    U_mesh = ∏(U_i + ΔU_i) ≈ U_target + ε∑J_i·H_i

where J_i are Jacobian-like sensitivity operators that encode how local
errors propagate through the mesh.

References:
- Ideas document Section 1: "Mathematical Framing of Error Accumulation"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import warnings


@dataclass
class PerturbationResult:
    """
    Results from perturbation analysis.

    Attributes
    ----------
    U_ideal : np.ndarray
        Ideal (noiseless) unitary.
    U_perturbed : np.ndarray
        Perturbed unitary with noise.
    delta_U : np.ndarray
        Perturbation matrix: U_perturbed - U_ideal.
    perturbation_norm : float
        Frobenius norm of δU.
    relative_error : float
        ||δU|| / ||U_ideal||.
    spectral_perturbation : float
        Spectral (operator) norm of δU.
    fidelity_loss : float
        1 - |⟨U_ideal|U_perturbed⟩|² / N².
    """
    U_ideal: np.ndarray
    U_perturbed: np.ndarray
    delta_U: np.ndarray
    perturbation_norm: float
    relative_error: float
    spectral_perturbation: float
    fidelity_loss: float


class PerturbationAnalyzer:
    """
    Analyzer for perturbation effects in photonic meshes.

    Implements first-order perturbation theory to understand how
    phase errors accumulate through the mesh.

    Parameters
    ----------
    mesh : PhotonicMesh
        The photonic mesh to analyze.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.n_modes = mesh.n_modes

    def analyze(self, include_higher_order: bool = False) -> PerturbationResult:
        """
        Analyze perturbation effects.

        Parameters
        ----------
        include_higher_order : bool
            If True, include second-order perturbation terms.

        Returns
        -------
        PerturbationResult
            Analysis results.
        """
        U_ideal = self.mesh.unitary(include_noise=False)
        U_perturbed = self.mesh.unitary(include_noise=True)
        delta_U = U_perturbed - U_ideal

        perturbation_norm = np.linalg.norm(delta_U, 'fro')
        ideal_norm = np.linalg.norm(U_ideal, 'fro')
        relative_error = perturbation_norm / ideal_norm if ideal_norm > 0 else 0

        # Spectral (operator) norm
        spectral_perturbation = np.linalg.norm(delta_U, 2)

        # Fidelity loss: E[1 - |⟨ψ|U†_mesh U_target|ψ⟩|²]
        # For unitaries, this simplifies to trace distance
        overlap = np.trace(U_ideal.conj().T @ U_perturbed)
        fidelity = np.abs(overlap) ** 2 / (self.n_modes ** 2)
        fidelity_loss = 1 - fidelity

        return PerturbationResult(
            U_ideal=U_ideal,
            U_perturbed=U_perturbed,
            delta_U=delta_U,
            perturbation_norm=perturbation_norm,
            relative_error=relative_error,
            spectral_perturbation=spectral_perturbation,
            fidelity_loss=fidelity_loss
        )

    def first_order_approximation(self) -> np.ndarray:
        """
        Compute first-order perturbation approximation.

        δU ≈ ∑_j U_N ⋯ δU_j ⋯ U_1

        Returns
        -------
        np.ndarray
            First-order approximation of δU.
        """
        n = self.n_modes
        delta_U_approx = np.zeros((n, n), dtype=np.complex128)

        # Compute prefix and suffix products for each layer
        n_layers = len(self.mesh.layers)
        prefixes = [np.eye(n, dtype=np.complex128)]
        suffixes = [np.eye(n, dtype=np.complex128)]

        # Forward pass: compute prefix products U_1, U_2·U_1, ...
        for layer in self.mesh.layers:
            U_layer = layer.unitary(include_noise=False)
            prefixes.append(U_layer @ prefixes[-1])

        # Backward pass: compute suffix products ..., U_L·U_{L-1}, U_L
        for layer in reversed(self.mesh.layers):
            U_layer = layer.unitary(include_noise=False)
            suffixes.append(suffixes[-1] @ U_layer)
        suffixes = suffixes[::-1]

        # Compute first-order contribution from each layer
        for j, layer in enumerate(self.mesh.layers):
            U_layer_ideal = layer.unitary(include_noise=False)
            U_layer_noisy = layer.unitary(include_noise=True)
            delta_U_layer = U_layer_noisy - U_layer_ideal

            # δU contribution: suffix @ δU_j @ prefix
            contribution = suffixes[j + 1] @ delta_U_layer @ prefixes[j]
            delta_U_approx += contribution

        return delta_U_approx

    def compute_layer_contributions(self) -> List[np.ndarray]:
        """
        Compute the perturbation contribution from each layer.

        Returns
        -------
        list of np.ndarray
            Perturbation contribution from each layer.
        """
        n = self.n_modes
        contributions = []

        # Compute prefix and suffix products
        n_layers = len(self.mesh.layers)
        prefixes = [np.eye(n, dtype=np.complex128)]
        suffixes = [np.eye(n, dtype=np.complex128)]

        for layer in self.mesh.layers:
            U_layer = layer.unitary(include_noise=False)
            prefixes.append(U_layer @ prefixes[-1])

        for layer in reversed(self.mesh.layers):
            U_layer = layer.unitary(include_noise=False)
            suffixes.append(suffixes[-1] @ U_layer)
        suffixes = suffixes[::-1]

        # Compute each layer's contribution
        for j, layer in enumerate(self.mesh.layers):
            U_layer_ideal = layer.unitary(include_noise=False)
            U_layer_noisy = layer.unitary(include_noise=True)
            delta_U_layer = U_layer_noisy - U_layer_ideal

            contribution = suffixes[j + 1] @ delta_U_layer @ prefixes[j]
            contributions.append(contribution)

        return contributions

    def error_localization(self, threshold: float = 0.1) -> Dict[str, any]:
        """
        Identify which MZIs contribute most to the total error.

        Parameters
        ----------
        threshold : float
            Fraction of total error to identify dominant contributors.

        Returns
        -------
        dict
            Dictionary with:
            - 'dominant_mzis': indices of MZIs contributing > threshold
            - 'contributions': normalized contribution of each MZI
            - 'cumulative': cumulative contribution
        """
        contributions = []

        for i, mzi in enumerate(self.mesh.mzis):
            # Compute contribution of this single MZI
            delta_theta = mzi.delta_theta
            delta_phi = mzi.delta_phi

            # Temporarily isolate this MZI's noise
            for other_mzi in self.mesh.mzis:
                other_mzi.delta_theta = 0
                other_mzi.delta_phi = 0

            mzi.delta_theta = delta_theta
            mzi.delta_phi = delta_phi

            result = self.analyze()
            contributions.append(result.perturbation_norm)

            # Restore
            mzi.delta_theta = 0
            mzi.delta_phi = 0

        # Restore all noise
        for i, mzi in enumerate(self.mesh.mzis):
            mzi.delta_theta = mzi.delta_theta
            mzi.delta_phi = mzi.delta_phi

        contributions = np.array(contributions)
        total = np.sum(contributions)
        if total > 0:
            normalized = contributions / total
        else:
            normalized = contributions

        # Sort by contribution
        sorted_indices = np.argsort(normalized)[::-1]
        cumulative = np.cumsum(normalized[sorted_indices])

        # Find dominant MZIs
        dominant_mask = cumulative <= (1 - threshold)
        n_dominant = np.sum(dominant_mask) + 1
        dominant_mzis = sorted_indices[:n_dominant]

        return {
            'dominant_mzis': dominant_mzis,
            'contributions': normalized,
            'cumulative': cumulative,
            'sorted_indices': sorted_indices
        }


def scaling_analysis(mesh_class, n_modes_range: List[int],
                     noise_model, n_samples: int = 100,
                     rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """
    Analyze how perturbation errors scale with mesh size.

    This implements the scaling law analysis: Error ∝ L·ε

    Parameters
    ----------
    mesh_class : class
        PhotonicMesh subclass to instantiate.
    n_modes_range : list of int
        Mesh sizes to test.
    noise_model : NoiseModel
        Noise model to apply.
    n_samples : int
        Number of Monte Carlo samples per size.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        Dictionary with:
        - 'n_modes': array of mesh sizes
        - 'mean_error': mean perturbation norm
        - 'std_error': standard deviation
        - 'n_mzis': number of MZIs per mesh
    """
    if rng is None:
        rng = np.random.default_rng()

    results = {
        'n_modes': [],
        'mean_error': [],
        'std_error': [],
        'n_mzis': [],
        'mean_fidelity_loss': [],
    }

    for n in n_modes_range:
        errors = []
        fidelity_losses = []

        mesh = mesh_class(n)
        results['n_mzis'].append(mesh.n_mzis)

        # Random phases for the mesh
        thetas = rng.uniform(0, np.pi / 2, mesh.n_mzis)
        phis = rng.uniform(0, 2 * np.pi, mesh.n_mzis)
        mesh.set_phases(thetas, phis)

        for _ in range(n_samples):
            # Apply noise
            noise_model.apply_to_mesh(mesh, rng)

            # Analyze
            analyzer = PerturbationAnalyzer(mesh)
            result = analyzer.analyze()

            errors.append(result.perturbation_norm)
            fidelity_losses.append(result.fidelity_loss)

            # Clear noise for next sample
            mesh.clear_noise()

        results['n_modes'].append(n)
        results['mean_error'].append(np.mean(errors))
        results['std_error'].append(np.std(errors))
        results['mean_fidelity_loss'].append(np.mean(fidelity_losses))

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    return results


def fit_scaling_law(n_mzis: np.ndarray, errors: np.ndarray) -> Tuple[float, float]:
    """
    Fit the scaling law: Error ≈ α · L^β

    Parameters
    ----------
    n_mzis : np.ndarray
        Number of MZIs (circuit depth proxy).
    errors : np.ndarray
        Mean errors.

    Returns
    -------
    alpha : float
        Scaling coefficient.
    beta : float
        Scaling exponent (expect ~0.5-1.0).
    """
    # Log-linear fit
    log_L = np.log(n_mzis)
    log_E = np.log(errors + 1e-15)

    # Linear regression
    coeffs = np.polyfit(log_L, log_E, 1)
    beta = coeffs[0]
    alpha = np.exp(coeffs[1])

    return alpha, beta
