"""
Mach-Zehnder Interferometer (MZI) unitary operators.

An MZI is the fundamental building block of programmable photonic meshes.
Each MZI implements a 2x2 unitary transformation parameterized by two
phase shifts: θ (internal) and φ (external).

The MZI unitary is given by:
    U(θ, φ) = [[cos(θ), i·sin(θ)], [i·sin(θ), cos(θ)]] · [[e^(iφ), 0], [0, 1]]

This decomposition allows any 2x2 unitary (up to global phase) to be realized.
"""

import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass, field


def mzi_unitary(theta: float, phi: float) -> np.ndarray:
    """
    Compute the 2x2 unitary matrix for a single MZI.

    The MZI transformation is:
        U(θ, φ) = R(θ) · P(φ)

    where R(θ) is the beam splitter rotation and P(φ) is the phase shift.

    Parameters
    ----------
    theta : float
        Internal phase shift (beam splitter angle), in radians.
        θ = 0 → bar state (full transmission)
        θ = π/2 → cross state (full coupling)
    phi : float
        External phase shift, in radians.

    Returns
    -------
    np.ndarray
        2x2 complex unitary matrix.

    Examples
    --------
    >>> U = mzi_unitary(np.pi/4, 0)  # 50:50 beam splitter
    >>> np.allclose(U @ U.conj().T, np.eye(2))  # Verify unitarity
    True
    """
    # Beam splitter (rotation) matrix
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, 1j * s],
        [1j * s, c]
    ], dtype=np.complex128)

    # Phase shift matrix
    P = np.array([
        [np.exp(1j * phi), 0],
        [0, 1]
    ], dtype=np.complex128)

    return R @ P


def mzi_unitary_symmetric(theta: float, phi_upper: float, phi_lower: float) -> np.ndarray:
    """
    Compute MZI unitary with symmetric phase shifters on both arms.

    This parameterization is sometimes used in Clements decomposition:
        U = P_upper · R(θ) · P_lower

    Parameters
    ----------
    theta : float
        Beam splitter angle.
    phi_upper : float
        Phase shift on upper arm.
    phi_lower : float
        Phase shift on lower arm.

    Returns
    -------
    np.ndarray
        2x2 complex unitary matrix.
    """
    c, s = np.cos(theta), np.sin(theta)

    return np.array([
        [np.exp(1j * phi_upper) * c, 1j * np.exp(1j * phi_upper) * s],
        [1j * np.exp(1j * phi_lower) * s, np.exp(1j * phi_lower) * c]
    ], dtype=np.complex128)


@dataclass
class MZI:
    """
    A Mach-Zehnder Interferometer with tunable phases.

    Attributes
    ----------
    theta : float
        Internal phase shift (beam splitter angle).
    phi : float
        External phase shift.
    mode_indices : Tuple[int, int]
        The two waveguide modes this MZI couples.
    label : str, optional
        Human-readable label for the MZI.

    Examples
    --------
    >>> mzi = MZI(theta=np.pi/4, phi=0, mode_indices=(0, 1))
    >>> U = mzi.unitary()
    >>> print(mzi.is_unitary())
    True
    """
    theta: float
    phi: float
    mode_indices: Tuple[int, int] = (0, 1)
    label: Optional[str] = None

    # Noise/perturbation fields
    delta_theta: float = field(default=0.0, repr=False)
    delta_phi: float = field(default=0.0, repr=False)

    def unitary(self, include_noise: bool = False) -> np.ndarray:
        """
        Compute the 2x2 unitary matrix for this MZI.

        Parameters
        ----------
        include_noise : bool
            If True, include phase perturbations (delta_theta, delta_phi).

        Returns
        -------
        np.ndarray
            2x2 complex unitary matrix.
        """
        if include_noise:
            return mzi_unitary(
                self.theta + self.delta_theta,
                self.phi + self.delta_phi
            )
        return mzi_unitary(self.theta, self.phi)

    def embed(self, n_modes: int, include_noise: bool = False) -> np.ndarray:
        """
        Embed the 2x2 MZI unitary into an n_modes × n_modes matrix.

        The MZI acts on modes specified by mode_indices, leaving
        other modes unchanged (identity).

        Parameters
        ----------
        n_modes : int
            Total number of modes in the system.
        include_noise : bool
            If True, include phase perturbations.

        Returns
        -------
        np.ndarray
            n_modes × n_modes unitary matrix.
        """
        i, j = self.mode_indices
        if i >= n_modes or j >= n_modes:
            raise ValueError(f"Mode indices {self.mode_indices} exceed n_modes={n_modes}")

        U_full = np.eye(n_modes, dtype=np.complex128)
        U_2x2 = self.unitary(include_noise=include_noise)

        # Insert 2x2 block
        U_full[i, i] = U_2x2[0, 0]
        U_full[i, j] = U_2x2[0, 1]
        U_full[j, i] = U_2x2[1, 0]
        U_full[j, j] = U_2x2[1, 1]

        return U_full

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """Check if the MZI matrix is unitary."""
        U = self.unitary()
        return np.allclose(U @ U.conj().T, np.eye(2), atol=tol)

    def set_noise(self, delta_theta: float = 0.0, delta_phi: float = 0.0):
        """Set phase noise perturbations."""
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi

    def clear_noise(self):
        """Clear all phase noise."""
        self.delta_theta = 0.0
        self.delta_phi = 0.0

    def perturbed_unitary(self) -> np.ndarray:
        """Get the unitary with current noise included."""
        return self.unitary(include_noise=True)

    def perturbation_matrix(self) -> np.ndarray:
        """
        Compute the perturbation ΔU = U_noisy - U_ideal.

        Returns
        -------
        np.ndarray
            2x2 perturbation matrix.
        """
        return self.unitary(include_noise=True) - self.unitary(include_noise=False)

    @property
    def transmission(self) -> float:
        """Power transmission (bar port) coefficient |t|²."""
        return np.cos(self.theta) ** 2

    @property
    def coupling(self) -> float:
        """Power coupling (cross port) coefficient |κ|²."""
        return np.sin(self.theta) ** 2

    def __repr__(self) -> str:
        label_str = f", label='{self.label}'" if self.label else ""
        return f"MZI(θ={self.theta:.4f}, φ={self.phi:.4f}, modes={self.mode_indices}{label_str})"


class MZILayer:
    """
    A layer of parallel MZIs acting on disjoint mode pairs.

    In mesh architectures, MZIs are often arranged in layers where
    each layer contains non-overlapping MZIs that can operate in parallel.

    Parameters
    ----------
    mzis : list of MZI
        List of MZI objects in this layer.
    n_modes : int
        Total number of modes.
    """

    def __init__(self, mzis: list, n_modes: int):
        self.mzis = mzis
        self.n_modes = n_modes
        self._validate()

    def _validate(self):
        """Ensure MZIs don't overlap."""
        used_modes = set()
        for mzi in self.mzis:
            i, j = mzi.mode_indices
            if i in used_modes or j in used_modes:
                raise ValueError(f"Overlapping MZIs detected at modes {mzi.mode_indices}")
            used_modes.add(i)
            used_modes.add(j)

    def unitary(self, include_noise: bool = False) -> np.ndarray:
        """
        Compute the unitary for this layer.

        Since MZIs are non-overlapping, the layer unitary is the
        product of individual embedded unitaries (order doesn't matter).

        Parameters
        ----------
        include_noise : bool
            If True, include phase perturbations.

        Returns
        -------
        np.ndarray
            n_modes × n_modes unitary matrix.
        """
        U = np.eye(self.n_modes, dtype=np.complex128)
        for mzi in self.mzis:
            U = mzi.embed(self.n_modes, include_noise=include_noise) @ U
        return U

    def __len__(self) -> int:
        return len(self.mzis)

    def __iter__(self):
        return iter(self.mzis)

    def __getitem__(self, idx):
        return self.mzis[idx]


# Utility functions for common MZI configurations

def balanced_splitter(mode_indices: Tuple[int, int] = (0, 1)) -> MZI:
    """Create a 50:50 beam splitter (θ = π/4)."""
    return MZI(theta=np.pi/4, phi=0, mode_indices=mode_indices, label="50:50")


def bar_state(mode_indices: Tuple[int, int] = (0, 1)) -> MZI:
    """Create a bar-state MZI (full transmission, θ = 0)."""
    return MZI(theta=0, phi=0, mode_indices=mode_indices, label="bar")


def cross_state(mode_indices: Tuple[int, int] = (0, 1)) -> MZI:
    """Create a cross-state MZI (full coupling, θ = π/2)."""
    return MZI(theta=np.pi/2, phi=0, mode_indices=mode_indices, label="cross")


def random_mzi(mode_indices: Tuple[int, int] = (0, 1), rng: Optional[np.random.Generator] = None) -> MZI:
    """
    Create an MZI with random phases uniformly distributed.

    θ ~ Uniform[0, π/2]
    φ ~ Uniform[0, 2π]
    """
    if rng is None:
        rng = np.random.default_rng()

    theta = rng.uniform(0, np.pi/2)
    phi = rng.uniform(0, 2*np.pi)
    return MZI(theta=theta, phi=phi, mode_indices=mode_indices)
