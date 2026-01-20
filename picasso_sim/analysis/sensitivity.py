"""
Sensitivity analysis and Jacobian computation for photonic meshes.

This module computes the sensitivity of the mesh unitary to phase
perturbations, implementing the Jacobian operators J_i from the
mathematical framework.

The key quantity is:
    ∂U/∂θ_j = U_N ⋯ U_{j+1} · (∂U_j/∂θ_j) · U_{j-1} ⋯ U_1

This enables:
1. Identification of sensitive vs. robust MZIs
2. Gradient computation for optimization
3. Uncertainty propagation analysis

References:
- Ideas document Section 4: "Operator-Theoretic Framing"
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SensitivityResult:
    """
    Results from sensitivity analysis.

    Attributes
    ----------
    jacobian_theta : np.ndarray
        Jacobian w.r.t. θ phases, shape (n_mzis, n, n) or flattened.
    jacobian_phi : np.ndarray
        Jacobian w.r.t. φ phases.
    sensitivity_norms : np.ndarray
        Frobenius norm of each Jacobian slice.
    max_sensitivity_idx : int
        Index of most sensitive MZI.
    condition_number : float
        Condition number of the Jacobian.
    """
    jacobian_theta: np.ndarray
    jacobian_phi: np.ndarray
    sensitivity_norms: np.ndarray
    max_sensitivity_idx: int
    condition_number: float


def mzi_derivative_theta(theta: float, phi: float) -> np.ndarray:
    """
    Compute ∂U/∂θ for a single MZI.

    U(θ, φ) = R(θ) · P(φ)
    ∂U/∂θ = (∂R/∂θ) · P(φ)

    Parameters
    ----------
    theta : float
        Internal phase shift.
    phi : float
        External phase shift.

    Returns
    -------
    np.ndarray
        2x2 derivative matrix.
    """
    c, s = np.cos(theta), np.sin(theta)

    # ∂R/∂θ
    dR_dtheta = np.array([
        [-s, 1j * c],
        [1j * c, -s]
    ], dtype=np.complex128)

    # P(φ)
    P = np.array([
        [np.exp(1j * phi), 0],
        [0, 1]
    ], dtype=np.complex128)

    return dR_dtheta @ P


def mzi_derivative_phi(theta: float, phi: float) -> np.ndarray:
    """
    Compute ∂U/∂φ for a single MZI.

    ∂U/∂φ = R(θ) · (∂P/∂φ)

    Parameters
    ----------
    theta : float
        Internal phase shift.
    phi : float
        External phase shift.

    Returns
    -------
    np.ndarray
        2x2 derivative matrix.
    """
    c, s = np.cos(theta), np.sin(theta)

    # R(θ)
    R = np.array([
        [c, 1j * s],
        [1j * s, c]
    ], dtype=np.complex128)

    # ∂P/∂φ
    dP_dphi = np.array([
        [1j * np.exp(1j * phi), 0],
        [0, 0]
    ], dtype=np.complex128)

    return R @ dP_dphi


def compute_jacobian(mesh, flatten: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the full Jacobian of the mesh unitary w.r.t. all phases.

    J_θ[j] = ∂U/∂θ_j = (∏_{k>j} U_k) · (∂U_j/∂θ_j) · (∏_{k<j} U_k)

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to analyze.
    flatten : bool
        If True, flatten to 2D Jacobian (n_mzis, n*n).

    Returns
    -------
    jacobian_theta : np.ndarray
        Shape (n_mzis, n, n) or (n_mzis, n*n).
    jacobian_phi : np.ndarray
        Same shape as jacobian_theta.
    """
    n = mesh.n_modes
    n_mzis = mesh.n_mzis

    # Compute prefix products: prefix[j] = U_{j-1} ⋯ U_1
    # and suffix products: suffix[j] = U_L ⋯ U_{j+1}
    prefixes = _compute_prefix_products(mesh)
    suffixes = _compute_suffix_products(mesh)

    jacobian_theta = np.zeros((n_mzis, n, n), dtype=np.complex128)
    jacobian_phi = np.zeros((n_mzis, n, n), dtype=np.complex128)

    mzi_idx = 0
    for layer_idx, layer in enumerate(mesh.layers):
        for mzi in layer.mzis:
            # Get MZI indices and parameters
            i, j = mzi.mode_indices
            theta, phi = mzi.theta, mzi.phi

            # Compute 2x2 derivatives
            dU_dtheta_2x2 = mzi_derivative_theta(theta, phi)
            dU_dphi_2x2 = mzi_derivative_phi(theta, phi)

            # Embed in n×n matrix
            dU_dtheta = np.zeros((n, n), dtype=np.complex128)
            dU_dphi = np.zeros((n, n), dtype=np.complex128)

            dU_dtheta[i, i] = dU_dtheta_2x2[0, 0]
            dU_dtheta[i, j] = dU_dtheta_2x2[0, 1]
            dU_dtheta[j, i] = dU_dtheta_2x2[1, 0]
            dU_dtheta[j, j] = dU_dtheta_2x2[1, 1]

            dU_dphi[i, i] = dU_dphi_2x2[0, 0]
            dU_dphi[i, j] = dU_dphi_2x2[0, 1]
            dU_dphi[j, i] = dU_dphi_2x2[1, 0]
            dU_dphi[j, j] = dU_dphi_2x2[1, 1]

            # Chain rule: ∂U/∂θ_j = suffix[j+1] · dU_j · prefix[j]
            jacobian_theta[mzi_idx] = suffixes[layer_idx + 1] @ dU_dtheta @ prefixes[layer_idx]
            jacobian_phi[mzi_idx] = suffixes[layer_idx + 1] @ dU_dphi @ prefixes[layer_idx]

            mzi_idx += 1

    if flatten:
        jacobian_theta = jacobian_theta.reshape(n_mzis, -1)
        jacobian_phi = jacobian_phi.reshape(n_mzis, -1)

    return jacobian_theta, jacobian_phi


def _compute_prefix_products(mesh) -> List[np.ndarray]:
    """Compute prefix products: prefix[j] = U_j · U_{j-1} ⋯ U_1."""
    n = mesh.n_modes
    prefixes = [np.eye(n, dtype=np.complex128)]

    for layer in mesh.layers:
        U_layer = layer.unitary(include_noise=False)
        prefixes.append(U_layer @ prefixes[-1])

    return prefixes


def _compute_suffix_products(mesh) -> List[np.ndarray]:
    """Compute suffix products: suffix[j] = U_L ⋯ U_{j+1}."""
    n = mesh.n_modes
    suffixes = [np.eye(n, dtype=np.complex128)]

    for layer in reversed(mesh.layers):
        U_layer = layer.unitary(include_noise=False)
        suffixes.append(suffixes[-1] @ U_layer)

    suffixes = suffixes[::-1]
    return suffixes


class SensitivityAnalyzer:
    """
    Analyzer for mesh sensitivity to phase perturbations.

    Computes Jacobians and identifies sensitive/robust regions.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.n_modes = mesh.n_modes
        self._jacobian_theta = None
        self._jacobian_phi = None

    def compute(self) -> SensitivityResult:
        """
        Compute full sensitivity analysis.

        Returns
        -------
        SensitivityResult
            Analysis results including Jacobians and sensitivity metrics.
        """
        J_theta, J_phi = compute_jacobian(self.mesh)
        self._jacobian_theta = J_theta
        self._jacobian_phi = J_phi

        # Compute sensitivity norms for each MZI
        n_mzis = self.mesh.n_mzis
        sensitivity_norms = np.zeros(n_mzis)

        for i in range(n_mzis):
            # Combined sensitivity from both phases
            norm_theta = np.linalg.norm(J_theta[i], 'fro')
            norm_phi = np.linalg.norm(J_phi[i], 'fro')
            sensitivity_norms[i] = np.sqrt(norm_theta**2 + norm_phi**2)

        max_idx = np.argmax(sensitivity_norms)

        # Condition number of flattened Jacobian
        J_combined = np.hstack([
            J_theta.reshape(n_mzis, -1),
            J_phi.reshape(n_mzis, -1)
        ])
        try:
            cond = np.linalg.cond(J_combined)
        except np.linalg.LinAlgError:
            cond = np.inf

        return SensitivityResult(
            jacobian_theta=J_theta,
            jacobian_phi=J_phi,
            sensitivity_norms=sensitivity_norms,
            max_sensitivity_idx=max_idx,
            condition_number=cond
        )

    def gradient_fidelity(self, U_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gradient of fidelity w.r.t. phase parameters.

        F = |Tr(U†_target U_mesh)|² / N²
        ∂F/∂θ_j = 2·Re[Tr(U†_target U_mesh)* · Tr(U†_target ∂U/∂θ_j)] / N²

        Parameters
        ----------
        U_target : np.ndarray
            Target unitary matrix.

        Returns
        -------
        grad_theta : np.ndarray
            Gradient w.r.t. θ phases.
        grad_phi : np.ndarray
            Gradient w.r.t. φ phases.
        """
        if self._jacobian_theta is None:
            self.compute()

        n = self.n_modes
        U_mesh = self.mesh.unitary()

        # Overlap
        overlap = np.trace(U_target.conj().T @ U_mesh)

        grad_theta = np.zeros(self.mesh.n_mzis)
        grad_phi = np.zeros(self.mesh.n_mzis)

        for i in range(self.mesh.n_mzis):
            # ∂overlap/∂θ_j
            d_overlap_theta = np.trace(U_target.conj().T @ self._jacobian_theta[i])
            d_overlap_phi = np.trace(U_target.conj().T @ self._jacobian_phi[i])

            # ∂F/∂θ_j
            grad_theta[i] = 2 * np.real(np.conj(overlap) * d_overlap_theta) / (n ** 2)
            grad_phi[i] = 2 * np.real(np.conj(overlap) * d_overlap_phi) / (n ** 2)

        return grad_theta, grad_phi

    def identify_robust_subgraph(self, threshold: float = 0.5) -> Dict[str, any]:
        """
        Identify MZIs with low sensitivity (robust subgraph).

        Parameters
        ----------
        threshold : float
            Fraction of max sensitivity below which MZIs are "robust".

        Returns
        -------
        dict
            'robust_indices': indices of low-sensitivity MZIs
            'sensitive_indices': indices of high-sensitivity MZIs
            'robustness_scores': 1 - normalized sensitivity
        """
        result = self.compute()
        norms = result.sensitivity_norms

        max_norm = np.max(norms)
        if max_norm > 0:
            normalized = norms / max_norm
        else:
            normalized = norms

        robust_mask = normalized < threshold
        sensitive_mask = ~robust_mask

        return {
            'robust_indices': np.where(robust_mask)[0],
            'sensitive_indices': np.where(sensitive_mask)[0],
            'robustness_scores': 1 - normalized,
            'sensitivity_map': normalized
        }

    def frechet_derivative(self, direction_theta: np.ndarray,
                          direction_phi: np.ndarray) -> np.ndarray:
        """
        Compute Fréchet derivative in a given direction.

        D_U[δθ, δφ] = ∑_j (∂U/∂θ_j)·δθ_j + (∂U/∂φ_j)·δφ_j

        Parameters
        ----------
        direction_theta : np.ndarray
            Direction vector for θ phases.
        direction_phi : np.ndarray
            Direction vector for φ phases.

        Returns
        -------
        np.ndarray
            Fréchet derivative (n×n matrix).
        """
        if self._jacobian_theta is None:
            self.compute()

        n = self.n_modes
        result = np.zeros((n, n), dtype=np.complex128)

        for i in range(self.mesh.n_mzis):
            result += direction_theta[i] * self._jacobian_theta[i]
            result += direction_phi[i] * self._jacobian_phi[i]

        return result

    def operator_norm_bound(self, noise_bound_theta: float,
                           noise_bound_phi: float) -> float:
        """
        Compute upper bound on ||δU|| given bounded noise.

        ||δU|| ≤ ∑_j ||J_θ[j]||·|δθ_j| + ||J_φ[j]||·|δφ_j|
               ≤ ε_θ·∑||J_θ|| + ε_φ·∑||J_φ||

        Parameters
        ----------
        noise_bound_theta : float
            Upper bound on |δθ_j| for all j.
        noise_bound_phi : float
            Upper bound on |δφ_j| for all j.

        Returns
        -------
        float
            Upper bound on perturbation norm.
        """
        if self._jacobian_theta is None:
            self.compute()

        sum_J_theta = sum(np.linalg.norm(J, 2) for J in self._jacobian_theta)
        sum_J_phi = sum(np.linalg.norm(J, 2) for J in self._jacobian_phi)

        return noise_bound_theta * sum_J_theta + noise_bound_phi * sum_J_phi


def adjoint_gradient(mesh, loss_fn, U_target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute gradient using adjoint method (efficient for large meshes).

    This is the photonic equivalent of backpropagation.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh.
    loss_fn : callable
        Loss function L(U) -> float.
    U_target : np.ndarray, optional
        Target unitary for fidelity-based loss.

    Returns
    -------
    grad_theta : np.ndarray
        Gradient w.r.t. θ.
    grad_phi : np.ndarray
        Gradient w.r.t. φ.
    """
    # Forward pass: compute all prefix products
    prefixes = _compute_prefix_products(mesh)
    U_mesh = prefixes[-1]

    # Compute loss gradient w.r.t. U_mesh
    if U_target is not None:
        # Fidelity loss: L = 1 - |Tr(U†U_target)|²/N²
        n = mesh.n_modes
        overlap = np.trace(U_target.conj().T @ U_mesh)
        dL_dU = -2 * np.conj(overlap) * U_target.conj().T / (n ** 2)
    else:
        # Numerical gradient
        eps = 1e-7
        L0 = loss_fn(U_mesh)
        dL_dU = np.zeros_like(U_mesh)
        for i in range(U_mesh.shape[0]):
            for j in range(U_mesh.shape[1]):
                U_mesh[i, j] += eps
                dL_dU[i, j] = (loss_fn(U_mesh) - L0) / eps
                U_mesh[i, j] -= eps

    # Backward pass: propagate gradient through layers
    grad_theta = np.zeros(mesh.n_mzis)
    grad_phi = np.zeros(mesh.n_mzis)

    adjoint = dL_dU.copy()
    mzi_idx = mesh.n_mzis - 1

    for layer_idx in range(len(mesh.layers) - 1, -1, -1):
        layer = mesh.layers[layer_idx]
        prefix = prefixes[layer_idx]

        for mzi in reversed(layer.mzis):
            i, j = mzi.mode_indices
            theta, phi = mzi.theta, mzi.phi

            # Compute local gradients
            dU_dtheta = mzi_derivative_theta(theta, phi)
            dU_dphi = mzi_derivative_phi(theta, phi)

            # Extract relevant 2x2 block of adjoint @ prefix†
            AP = adjoint @ prefix.conj().T
            block = AP[np.ix_([i, j], [i, j])]

            # Gradient contribution
            grad_theta[mzi_idx] = np.real(np.trace(block @ dU_dtheta.conj().T))
            grad_phi[mzi_idx] = np.real(np.trace(block @ dU_dphi.conj().T))

            mzi_idx -= 1

        # Update adjoint for next layer
        U_layer = layer.unitary()
        adjoint = U_layer.conj().T @ adjoint

    return grad_theta, grad_phi
