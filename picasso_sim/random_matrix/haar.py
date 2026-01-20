"""
Haar random unitary generation and analysis.

A Haar-distributed unitary is drawn uniformly from the unitary group U(N).
These serve as:
1. Benchmarks for universality of photonic meshes
2. Fault-tolerant initializations (error delocalization)
3. Random circuit analysis

Key insight from ideas document:
"Haar unitary matrices spread information uniformly across all input/output
modes. This delocalizes errors: even if a few MZIs are faulty, their
influence gets averaged out over the whole system."

References:
- Mezzadri, "How to generate random matrices from the classical compact groups"
"""

import numpy as np
from typing import Optional, Tuple
from scipy import stats


def haar_unitary(n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a Haar-random unitary matrix.

    Uses the QR decomposition method:
    1. Generate Z ~ CN(0, I) (complex Gaussian matrix)
    2. Compute QR decomposition Z = QR
    3. Normalize: U = Q · diag(R_ii / |R_ii|)

    Parameters
    ----------
    n : int
        Dimension of the unitary matrix.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        n×n Haar-random unitary matrix.

    Examples
    --------
    >>> U = haar_unitary(4)
    >>> np.allclose(U @ U.conj().T, np.eye(4))
    True
    """
    if rng is None:
        rng = np.random.default_rng()

    # Complex Gaussian matrix
    Z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Z /= np.sqrt(2)

    # QR decomposition
    Q, R = np.linalg.qr(Z)

    # Correct phases to ensure Haar measure
    d = np.diag(R)
    ph = d / np.abs(d)
    U = Q * ph  # Multiply columns by phases

    return U


def haar_unitary_batch(n: int, batch_size: int,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate a batch of Haar-random unitaries.

    Parameters
    ----------
    n : int
        Dimension.
    batch_size : int
        Number of unitaries to generate.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Shape (batch_size, n, n) array of unitaries.
    """
    if rng is None:
        rng = np.random.default_rng()

    return np.array([haar_unitary(n, rng) for _ in range(batch_size)])


def sample_haar_phases(n_modes: int, mesh_n_mzis: int,
                       rng: Optional[np.random.Generator] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample phases that would produce a Haar-random unitary on a mesh.

    Note: This is an approximation. True Haar sampling requires
    decomposing a Haar unitary into mesh parameters.

    Parameters
    ----------
    n_modes : int
        Number of modes.
    mesh_n_mzis : int
        Number of MZIs in the mesh.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    thetas : np.ndarray
        Internal phase shifts sampled appropriately.
    phis : np.ndarray
        External phase shifts.
    """
    if rng is None:
        rng = np.random.default_rng()

    # For Haar measure, θ should be sampled with appropriate distribution
    # For MZI: θ controls splitting ratio, uniform on [0, π/2] is NOT Haar
    # Correct distribution: P(θ) ∝ sin(2θ) on [0, π/2]

    # Inverse CDF sampling for sin(2θ) distribution
    u = rng.uniform(0, 1, mesh_n_mzis)
    thetas = 0.5 * np.arccos(1 - 2 * u)  # Maps to [0, π/2]

    # φ is uniform on [0, 2π]
    phis = rng.uniform(0, 2 * np.pi, mesh_n_mzis)

    return thetas, phis


def is_haar_distributed(unitaries: np.ndarray, significance: float = 0.05) -> dict:
    """
    Statistical test for Haar distribution.

    Tests whether a collection of unitaries appears to be drawn
    from the Haar measure using several statistics.

    Parameters
    ----------
    unitaries : np.ndarray
        Shape (n_samples, n, n) array of unitary matrices.
    significance : float
        Significance level for hypothesis tests.

    Returns
    -------
    dict
        Test results including:
        - 'eigenvalue_test': Circular ensemble test on eigenvalues
        - 'entry_test': Test that entries are uniform on disk
        - 'is_haar': Overall conclusion
    """
    n_samples, n, _ = unitaries.shape

    results = {}

    # Test 1: Eigenvalue statistics
    # Eigenvalues should be uniform on unit circle (CUE)
    all_phases = []
    for U in unitaries:
        eigs = np.linalg.eigvals(U)
        phases = np.angle(eigs)
        all_phases.extend(phases)

    all_phases = np.array(all_phases)
    # Normalize to [0, 1] and test uniformity
    normalized_phases = (all_phases + np.pi) / (2 * np.pi)
    ks_stat, ks_pvalue = stats.kstest(normalized_phases, 'uniform')
    results['eigenvalue_ks_pvalue'] = ks_pvalue
    results['eigenvalue_test_passed'] = ks_pvalue > significance

    # Test 2: Matrix entry statistics
    # Real and imaginary parts of U[i,j] should be approximately Gaussian
    # with variance 1/(2n)
    entries = unitaries[:, 0, 0]  # Sample (0,0) entry
    real_parts = np.real(entries)
    imag_parts = np.imag(entries)

    expected_std = 1 / np.sqrt(2 * n)

    # Test variance
    real_std = np.std(real_parts)
    imag_std = np.std(imag_parts)
    results['expected_std'] = expected_std
    results['observed_real_std'] = real_std
    results['observed_imag_std'] = imag_std

    # Chi-squared test for variance
    std_ratio = (real_std / expected_std)
    results['entry_test_passed'] = 0.5 < std_ratio < 2.0

    # Overall conclusion
    results['is_haar'] = (results['eigenvalue_test_passed'] and
                          results['entry_test_passed'])

    return results


def circular_unitary_ensemble_spacing(n: int, n_samples: int = 1000,
                                      rng: Optional[np.random.Generator] = None
                                      ) -> np.ndarray:
    """
    Sample eigenvalue spacing distribution from CUE.

    For Haar unitaries, consecutive eigenvalue spacings follow
    the Wigner surmise (approximately).

    Parameters
    ----------
    n : int
        Matrix dimension.
    n_samples : int
        Number of samples.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Normalized eigenvalue spacings.
    """
    if rng is None:
        rng = np.random.default_rng()

    all_spacings = []

    for _ in range(n_samples):
        U = haar_unitary(n, rng)
        eigs = np.linalg.eigvals(U)
        phases = np.sort(np.angle(eigs))

        # Compute spacings (including wrap-around)
        spacings = np.diff(phases)
        spacings = np.append(spacings, 2*np.pi + phases[0] - phases[-1])

        # Normalize by mean spacing
        mean_spacing = 2 * np.pi / n
        normalized = spacings / mean_spacing
        all_spacings.extend(normalized)

    return np.array(all_spacings)


def wigner_surmise(s: np.ndarray, beta: int = 2) -> np.ndarray:
    """
    Wigner surmise for eigenvalue spacing distribution.

    P(s) = a_β · s^β · exp(-b_β · s²)

    Parameters
    ----------
    s : np.ndarray
        Spacing values (normalized).
    beta : int
        Dyson index (2 for CUE/GUE).

    Returns
    -------
    np.ndarray
        Probability density at each s.
    """
    if beta == 1:  # COE/GOE
        a = np.pi / 2
        b = np.pi / 4
    elif beta == 2:  # CUE/GUE
        a = 32 / np.pi**2
        b = 4 / np.pi
    elif beta == 4:  # CSE/GSE
        a = 2**18 / (3**6 * np.pi**3)
        b = 64 / (9 * np.pi)
    else:
        raise ValueError(f"Invalid beta: {beta}")

    return a * s**beta * np.exp(-b * s**2)


def haar_fidelity_benchmark(mesh, n_samples: int = 100,
                           rng: Optional[np.random.Generator] = None) -> dict:
    """
    Benchmark mesh's ability to implement Haar-random unitaries.

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to test.
    n_samples : int
        Number of Haar unitaries to test.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        'fidelities': array of achieved fidelities
        'mean_fidelity': average fidelity
        'min_fidelity': worst-case fidelity
        'implementation_rate': fraction achieving F > 0.99
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        from ..analysis.fidelity import fidelity
    except ImportError:
        from picasso_sim.analysis.fidelity import fidelity

    fidelities = []

    for _ in range(n_samples):
        # Generate target Haar unitary
        U_target = haar_unitary(mesh.n_modes, rng)

        # Set mesh to implement it (using random phases as approximation)
        thetas, phis = sample_haar_phases(mesh.n_modes, mesh.n_mzis, rng)
        mesh.set_phases(thetas, phis)

        U_mesh = mesh.unitary()
        fid = fidelity(U_target, U_mesh)
        fidelities.append(fid)

    fidelities = np.array(fidelities)

    return {
        'fidelities': fidelities,
        'mean_fidelity': np.mean(fidelities),
        'std_fidelity': np.std(fidelities),
        'min_fidelity': np.min(fidelities),
        'max_fidelity': np.max(fidelities),
        'implementation_rate_99': np.mean(fidelities > 0.99),
        'implementation_rate_999': np.mean(fidelities > 0.999)
    }


def error_delocalization_test(mesh, noise_model, n_samples: int = 100,
                              rng: Optional[np.random.Generator] = None) -> dict:
    """
    Test error delocalization property of Haar-random meshes.

    Compare error amplification for:
    1. Structured (identity) configuration
    2. Random (Haar-like) configuration

    Parameters
    ----------
    mesh : PhotonicMesh
        The mesh to test.
    noise_model : NoiseModel
        Noise model to apply.
    n_samples : int
        Number of samples.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    dict
        Comparison of error behavior.
    """
    if rng is None:
        rng = np.random.default_rng()

    try:
        from ..analysis.fidelity import fidelity
    except ImportError:
        from picasso_sim.analysis.fidelity import fidelity

    # Test 1: Identity-like configuration (all θ ≈ 0)
    mesh.set_phases(
        np.zeros(mesh.n_mzis),
        np.zeros(mesh.n_mzis)
    )
    U_identity = mesh.unitary()

    identity_fidelities = []
    for _ in range(n_samples):
        noise_model.apply_to_mesh(mesh, rng)
        U_noisy = mesh.unitary(include_noise=True)
        identity_fidelities.append(fidelity(U_identity, U_noisy))
        mesh.clear_noise()

    # Test 2: Random (Haar-like) configuration
    thetas, phis = sample_haar_phases(mesh.n_modes, mesh.n_mzis, rng)
    mesh.set_phases(thetas, phis)
    U_random = mesh.unitary()

    random_fidelities = []
    for _ in range(n_samples):
        noise_model.apply_to_mesh(mesh, rng)
        U_noisy = mesh.unitary(include_noise=True)
        random_fidelities.append(fidelity(U_random, U_noisy))
        mesh.clear_noise()

    return {
        'identity_mean_fidelity': np.mean(identity_fidelities),
        'identity_std_fidelity': np.std(identity_fidelities),
        'random_mean_fidelity': np.mean(random_fidelities),
        'random_std_fidelity': np.std(random_fidelities),
        'delocalization_benefit': np.mean(random_fidelities) - np.mean(identity_fidelities)
    }
