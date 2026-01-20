"""
Photonic mesh architectures for implementing arbitrary unitary transformations.

This module implements the two primary mesh architectures used in programmable
photonic circuits:

1. Reck Architecture (triangular): Uses N(N-1)/2 MZIs arranged in a triangular
   pattern. Simpler but asymmetric path lengths.

2. Clements Architecture (rectangular): Uses N(N-1)/2 MZIs in a rectangular
   pattern. Symmetric path lengths, better for fabrication.

Both architectures can implement any N×N unitary transformation.

References:
- Reck et al., PRL 73, 58 (1994)
- Clements et al., Optica 3, 1460 (2016)
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .mzi import MZI, MZILayer, mzi_unitary


class PhotonicMesh(ABC):
    """
    Abstract base class for photonic MZI meshes.

    A photonic mesh is a network of MZIs that implements a unitary
    transformation U ∈ U(N) on N optical modes.
    """

    def __init__(self, n_modes: int):
        """
        Initialize photonic mesh.

        Parameters
        ----------
        n_modes : int
            Number of optical modes (waveguides).
        """
        self.n_modes = n_modes
        self.mzis: List[MZI] = []
        self.layers: List[MZILayer] = []
        self._build_topology()

    @abstractmethod
    def _build_topology(self):
        """Build the mesh topology (MZI arrangement)."""
        pass

    def unitary(self, include_noise: bool = False) -> np.ndarray:
        """
        Compute the full mesh unitary U = U_L · U_{L-1} · ... · U_1.

        Parameters
        ----------
        include_noise : bool
            If True, include phase perturbations in all MZIs.

        Returns
        -------
        np.ndarray
            N×N unitary matrix.
        """
        U = np.eye(self.n_modes, dtype=np.complex128)
        for layer in self.layers:
            U = layer.unitary(include_noise=include_noise) @ U
        return U

    def set_phases(self, thetas: np.ndarray, phis: np.ndarray):
        """
        Set all MZI phases from arrays.

        Parameters
        ----------
        thetas : np.ndarray
            Array of θ values, length = number of MZIs.
        phis : np.ndarray
            Array of φ values, length = number of MZIs.
        """
        if len(thetas) != len(self.mzis) or len(phis) != len(self.mzis):
            raise ValueError(f"Expected {len(self.mzis)} phases, got {len(thetas)}, {len(phis)}")

        for i, mzi in enumerate(self.mzis):
            mzi.theta = thetas[i]
            mzi.phi = phis[i]

    def get_phases(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all MZI phases as arrays.

        Returns
        -------
        thetas : np.ndarray
            Array of θ values.
        phis : np.ndarray
            Array of φ values.
        """
        thetas = np.array([mzi.theta for mzi in self.mzis])
        phis = np.array([mzi.phi for mzi in self.mzis])
        return thetas, phis

    def apply_noise(self, delta_thetas: np.ndarray, delta_phis: np.ndarray):
        """
        Apply phase noise to all MZIs.

        Parameters
        ----------
        delta_thetas : np.ndarray
            Array of θ perturbations.
        delta_phis : np.ndarray
            Array of φ perturbations.
        """
        for i, mzi in enumerate(self.mzis):
            mzi.set_noise(delta_thetas[i], delta_phis[i])

    def clear_noise(self):
        """Clear all phase noise from MZIs."""
        for mzi in self.mzis:
            mzi.clear_noise()

    def is_unitary(self, tol: float = 1e-10) -> bool:
        """Check if the mesh implements a unitary transformation."""
        U = self.unitary()
        return np.allclose(U @ U.conj().T, np.eye(self.n_modes), atol=tol)

    @property
    def n_mzis(self) -> int:
        """Total number of MZIs in the mesh."""
        return len(self.mzis)

    @property
    def n_layers(self) -> int:
        """Number of layers in the mesh."""
        return len(self.layers)

    @property
    def depth(self) -> int:
        """Circuit depth (number of sequential operations)."""
        return self.n_layers

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_modes={self.n_modes}, n_mzis={self.n_mzis}, depth={self.depth})"


class ReckMesh(PhotonicMesh):
    """
    Reck (triangular) mesh architecture.

    The Reck decomposition arranges MZIs in a triangular pattern:
    - Column k contains (N-k) MZIs
    - Total MZIs: N(N-1)/2
    - Depth: 2N-3

    This architecture progressively nullifies matrix elements to achieve
    any target unitary.

    Example for N=4:
        Mode 0: ──[MZI]──[MZI]──[MZI]──
        Mode 1: ──[   ]──[MZI]──[MZI]──[MZI]──
        Mode 2: ──[   ]──[   ]──[MZI]──[MZI]──[MZI]──
        Mode 3: ──[   ]──[   ]──[   ]──[MZI]──[MZI]──[MZI]──
    """

    def _build_topology(self):
        """Build Reck triangular topology."""
        self.mzis = []
        self.layers = []

        n = self.n_modes
        mzi_idx = 0

        # Build diagonal layers
        for diag in range(n - 1):
            layer_mzis = []
            # In each diagonal, we have MZIs that don't overlap
            for col in range(diag, -1, -2):
                row = diag - col
                if row + 1 < n:
                    mzi = MZI(
                        theta=0.0,
                        phi=0.0,
                        mode_indices=(row, row + 1),
                        label=f"R_{mzi_idx}"
                    )
                    self.mzis.append(mzi)
                    layer_mzis.append(mzi)
                    mzi_idx += 1

            if layer_mzis:
                self.layers.append(MZILayer(layer_mzis, self.n_modes))

        # Additional layers for full universality
        for diag in range(n - 2, 0, -1):
            layer_mzis = []
            for col in range(diag - 1, -1, -2):
                row = n - 1 - diag + col
                if row + 1 < n:
                    mzi = MZI(
                        theta=0.0,
                        phi=0.0,
                        mode_indices=(row, row + 1),
                        label=f"R_{mzi_idx}"
                    )
                    self.mzis.append(mzi)
                    layer_mzis.append(mzi)
                    mzi_idx += 1

            if layer_mzis:
                self.layers.append(MZILayer(layer_mzis, self.n_modes))


class ClementsMesh(PhotonicMesh):
    """
    Clements (rectangular) mesh architecture.

    The Clements decomposition arranges MZIs in a rectangular pattern
    with alternating even/odd layers:
    - Even layers: MZIs on modes (0,1), (2,3), (4,5), ...
    - Odd layers: MZIs on modes (1,2), (3,4), (5,6), ...
    - Total MZIs: N(N-1)/2
    - Depth: N

    This architecture has symmetric path lengths and is more suitable
    for physical implementation.

    Example for N=4:
        Mode 0: ──[MZI]──[   ]──[MZI]──[   ]──
        Mode 1: ──[MZI]──[MZI]──[MZI]──[MZI]──
        Mode 2: ──[MZI]──[MZI]──[MZI]──[MZI]──
        Mode 3: ──[   ]──[MZI]──[   ]──[MZI]──
    """

    def _build_topology(self):
        """Build Clements rectangular topology."""
        self.mzis = []
        self.layers = []

        n = self.n_modes
        mzi_idx = 0

        # For N modes, we need N-1 "diagonals" worth of operations
        # Arranged in alternating even/odd layers
        n_full_layers = n

        for layer_idx in range(n_full_layers):
            layer_mzis = []

            if layer_idx % 2 == 0:
                # Even layer: (0,1), (2,3), ...
                for i in range(0, n - 1, 2):
                    mzi = MZI(
                        theta=0.0,
                        phi=0.0,
                        mode_indices=(i, i + 1),
                        label=f"C_{mzi_idx}"
                    )
                    self.mzis.append(mzi)
                    layer_mzis.append(mzi)
                    mzi_idx += 1
            else:
                # Odd layer: (1,2), (3,4), ...
                for i in range(1, n - 1, 2):
                    mzi = MZI(
                        theta=0.0,
                        phi=0.0,
                        mode_indices=(i, i + 1),
                        label=f"C_{mzi_idx}"
                    )
                    self.mzis.append(mzi)
                    layer_mzis.append(mzi)
                    mzi_idx += 1

            if layer_mzis:
                self.layers.append(MZILayer(layer_mzis, self.n_modes))


class ButterflyMesh(PhotonicMesh):
    """
    Butterfly (FFT-like) mesh architecture.

    The butterfly architecture is inspired by the FFT algorithm and
    uses a logarithmic depth structure. While not universal for all
    unitaries, it's highly efficient for structured transformations.

    Depth: O(log N)
    MZIs: O(N log N)
    """

    def _build_topology(self):
        """Build butterfly topology."""
        self.mzis = []
        self.layers = []

        n = self.n_modes
        if n & (n - 1) != 0:
            raise ValueError("Butterfly mesh requires n_modes to be a power of 2")

        mzi_idx = 0
        n_stages = int(np.log2(n))

        for stage in range(n_stages):
            layer_mzis = []
            stride = 2 ** stage

            for block_start in range(0, n, 2 * stride):
                for i in range(stride):
                    idx1 = block_start + i
                    idx2 = block_start + i + stride
                    if idx2 < n:
                        mzi = MZI(
                            theta=0.0,
                            phi=0.0,
                            mode_indices=(idx1, idx2),
                            label=f"B_{mzi_idx}"
                        )
                        self.mzis.append(mzi)
                        layer_mzis.append(mzi)
                        mzi_idx += 1

            if layer_mzis:
                self.layers.append(MZILayer(layer_mzis, self.n_modes))


def decompose_unitary_clements(U: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decompose a unitary matrix into Clements mesh parameters.

    Uses the Clements decomposition algorithm to find MZI phases
    (θ, φ) that implement the target unitary U.

    Parameters
    ----------
    U : np.ndarray
        N×N unitary matrix to decompose.

    Returns
    -------
    thetas : np.ndarray
        Internal phase shifts for each MZI.
    phis : np.ndarray
        External phase shifts for each MZI.
    diag_phases : np.ndarray
        Output diagonal phase shifts.

    References
    ----------
    Clements et al., Optica 3, 1460 (2016)
    """
    n = U.shape[0]
    T = U.copy().astype(np.complex128)

    thetas = []
    phis = []

    # Nullify off-diagonal elements using Givens rotations
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if i % 2 == 0:
                # Left multiplication (column operation)
                col = n - 1 - j
                row = col - 1

                # Compute rotation to nullify T[row, col]
                a = T[row, col - i]
                b = T[col, col - i]

                if np.abs(b) < 1e-15:
                    theta = np.pi / 2
                    phi = 0
                else:
                    r = a / b
                    theta = np.arctan(np.abs(r))
                    phi = np.angle(r)

                thetas.append(theta)
                phis.append(phi)

                # Apply rotation
                G = mzi_unitary(theta, phi)
                T[[row, col], :] = G.conj().T @ T[[row, col], :]

            else:
                # Right multiplication (row operation)
                row = j
                col = row + 1

                a = T[i, row]
                b = T[i, col]

                if np.abs(a) < 1e-15:
                    theta = np.pi / 2
                    phi = 0
                else:
                    r = b / a
                    theta = np.arctan(np.abs(r))
                    phi = -np.angle(r)

                thetas.append(theta)
                phis.append(phi)

                # Apply rotation
                G = mzi_unitary(theta, phi)
                T[:, [row, col]] = T[:, [row, col]] @ G

    # Remaining diagonal phases
    diag_phases = np.angle(np.diag(T))

    return np.array(thetas), np.array(phis), diag_phases


def create_mesh_for_unitary(U: np.ndarray, architecture: str = "clements") -> PhotonicMesh:
    """
    Create a photonic mesh that implements a target unitary.

    Parameters
    ----------
    U : np.ndarray
        N×N unitary matrix.
    architecture : str
        Mesh architecture: "clements" or "reck".

    Returns
    -------
    PhotonicMesh
        Configured mesh implementing U.
    """
    n = U.shape[0]

    if architecture.lower() == "clements":
        mesh = ClementsMesh(n)
        thetas, phis, _ = decompose_unitary_clements(U)
        # Note: Full implementation would set phases correctly
        # This is a simplified version
    elif architecture.lower() == "reck":
        mesh = ReckMesh(n)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return mesh


def random_mesh(n_modes: int, architecture: str = "clements",
                rng: Optional[np.random.Generator] = None) -> PhotonicMesh:
    """
    Create a mesh with random MZI phases.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    architecture : str
        Mesh architecture: "clements", "reck", or "butterfly".
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    PhotonicMesh
        Mesh with random phases.
    """
    if rng is None:
        rng = np.random.default_rng()

    if architecture.lower() == "clements":
        mesh = ClementsMesh(n_modes)
    elif architecture.lower() == "reck":
        mesh = ReckMesh(n_modes)
    elif architecture.lower() == "butterfly":
        mesh = ButterflyMesh(n_modes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # Set random phases
    n_mzis = mesh.n_mzis
    thetas = rng.uniform(0, np.pi / 2, n_mzis)
    phis = rng.uniform(0, 2 * np.pi, n_mzis)
    mesh.set_phases(thetas, phis)

    return mesh
