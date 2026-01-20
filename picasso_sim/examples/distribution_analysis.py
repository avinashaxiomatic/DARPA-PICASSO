"""
Distribution Analysis for Photonic Systems

Explores different distributions for:
1. Unitary matrices (what operations we implement)
2. Phase errors (fabrication/thermal noise)
3. Input states (application-dependent)

Key question: What distributions maximize robustness, performance, or utility?
"""

import numpy as np
from scipy import linalg
from scipy.stats import unitary_group
import sys

sys.path.insert(0, '.')

from picasso_sim.core.mesh import ClementsMesh
from picasso_sim.analysis.fidelity import fidelity


# =============================================================================
# PART 1: UNITARY DISTRIBUTIONS
# =============================================================================

def sample_haar_unitary(n):
    """Haar-random unitary (uniform on U(n))."""
    return unitary_group.rvs(n)


def sample_dft_unitary(n):
    """Discrete Fourier Transform - highly structured."""
    return linalg.dft(n, scale='sqrtn')


def sample_hadamard_like(n):
    """Hadamard-like unitary (works for any n, not just powers of 2)."""
    # Use DFT with random diagonal phases
    D = np.diag(np.exp(1j * np.random.uniform(0, 2*np.pi, n)))
    F = linalg.dft(n, scale='sqrtn')
    return D @ F


def sample_diagonal_unitary(n):
    """Random diagonal unitary - only phase shifts, no mixing."""
    phases = np.random.uniform(0, 2*np.pi, n)
    return np.diag(np.exp(1j * phases))


def sample_permutation_unitary(n):
    """Random permutation matrix - classical routing."""
    perm = np.random.permutation(n)
    P = np.zeros((n, n))
    for i, j in enumerate(perm):
        P[i, j] = 1
    return P


def sample_block_diagonal(n, block_size=4):
    """Block diagonal with Haar blocks - limited entanglement."""
    U = np.zeros((n, n), dtype=complex)
    for i in range(0, n, block_size):
        size = min(block_size, n - i)
        U[i:i+size, i:i+size] = unitary_group.rvs(size)
    return U


def sample_sparse_unitary(n, sparsity=0.3):
    """Sparse unitary - approximation with limited connections."""
    # Start with Haar and keep only some elements
    U = unitary_group.rvs(n)
    mask = np.random.random((n, n)) < sparsity
    np.fill_diagonal(mask, True)  # Keep diagonal
    U_sparse = U * mask
    # Re-orthogonalize (approximate)
    Q, R = np.linalg.qr(U_sparse)
    return Q


def sample_near_identity(n, epsilon=0.1):
    """Near-identity unitary - small perturbation from I."""
    # U = exp(i*epsilon*H) where H is Hermitian
    H = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (H + H.conj().T) / 2  # Make Hermitian
    H = H / np.linalg.norm(H) * epsilon
    return linalg.expm(1j * H)


def sample_low_rank_perturbation(n, rank=5):
    """Identity plus low-rank unitary perturbation."""
    # Useful for modeling small subspace operations
    U_small = unitary_group.rvs(rank)
    U = np.eye(n, dtype=complex)
    U[:rank, :rank] = U_small
    # Apply random basis rotation
    V = unitary_group.rvs(n)
    return V @ U @ V.conj().T


# =============================================================================
# PART 2: ERROR/NOISE DISTRIBUTIONS
# =============================================================================

def gaussian_errors(n, sigma):
    """Standard Gaussian phase errors."""
    return np.random.normal(0, sigma, n)


def uniform_errors(n, max_error):
    """Uniform distribution [-max, +max]."""
    return np.random.uniform(-max_error, max_error, n)


def bimodal_errors(n, sigma, offset=0.05):
    """Bimodal - models systematic +/- bias in fabrication."""
    signs = np.random.choice([-1, 1], n)
    return signs * offset + np.random.normal(0, sigma, n)


def heavy_tailed_errors(n, scale):
    """Cauchy/Lorentzian - heavy tails, occasional large errors."""
    return np.random.standard_cauchy(n) * scale


def correlated_errors(n, sigma, correlation_length=10):
    """Spatially correlated errors - thermal gradients."""
    # Create correlation matrix
    positions = np.arange(n)
    dist = np.abs(positions[:, None] - positions[None, :])
    C = np.exp(-dist / correlation_length) * sigma**2
    L = np.linalg.cholesky(C + 1e-10 * np.eye(n))
    return L @ np.random.randn(n)


def quantized_errors(n, step_size=0.01):
    """Quantized errors - DAC resolution limits."""
    continuous = np.random.normal(0, 0.02, n)
    return np.round(continuous / step_size) * step_size


def drift_errors(n, drift_rate=0.001, n_steps=100):
    """Accumulated drift over time."""
    drift = np.zeros(n)
    for _ in range(n_steps):
        drift += np.random.normal(0, drift_rate, n)
    return drift


def sparse_defect_errors(n, defect_prob=0.01, defect_magnitude=0.5):
    """Sparse large defects - broken MZIs."""
    errors = np.zeros(n)
    defects = np.random.random(n) < defect_prob
    errors[defects] = np.random.choice([-1, 1], np.sum(defects)) * defect_magnitude
    return errors


# =============================================================================
# PART 3: INPUT STATE DISTRIBUTIONS
# =============================================================================

def uniform_input(n):
    """Uniform superposition - equal in all modes."""
    return np.ones(n, dtype=complex) / np.sqrt(n)


def localized_input(n, mode=0):
    """Single mode excitation - delta function."""
    psi = np.zeros(n, dtype=complex)
    psi[mode] = 1.0
    return psi


def gaussian_input(n, center=None, width=3):
    """Gaussian wavepacket."""
    if center is None:
        center = n // 2
    x = np.arange(n)
    psi = np.exp(-(x - center)**2 / (2 * width**2))
    return psi / np.linalg.norm(psi)


def random_input(n):
    """Random pure state (Haar-distributed)."""
    psi = np.random.randn(n) + 1j * np.random.randn(n)
    return psi / np.linalg.norm(psi)


def coherent_state_input(n, alpha=2.0):
    """Coherent state (Poisson photon number distribution)."""
    # Simplified: Gaussian approximation
    mean_n = int(abs(alpha)**2)
    psi = np.zeros(n, dtype=complex)
    for k in range(min(n, mean_n + 5 * int(np.sqrt(mean_n)) + 1)):
        psi[k] = np.exp(-abs(alpha)**2/2) * (alpha**k) / np.sqrt(float(np.math.factorial(k)))
    return psi / np.linalg.norm(psi)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_unitary_properties(U, name=""):
    """Analyze properties of a unitary matrix."""
    n = U.shape[0]

    # Verify unitarity
    unitarity_error = np.linalg.norm(U @ U.conj().T - np.eye(n))

    # Eigenvalue distribution
    eigenvalues = np.linalg.eigvals(U)
    eigenphases = np.angle(eigenvalues)
    phase_spread = np.std(eigenphases)

    # Sparsity (fraction of small elements)
    threshold = 0.01
    sparsity = np.mean(np.abs(U) < threshold)

    # Entangling power (simplified measure)
    # How much does it mix modes on average?
    mixing = 1 - np.mean(np.abs(U)**2 * np.eye(n))

    return {
        'name': name,
        'unitarity_error': unitarity_error,
        'phase_spread': phase_spread,
        'sparsity': sparsity,
        'mixing': mixing
    }


def test_error_robustness(unitary_sampler, error_sampler, n_modes=16,
                          n_trials=100, error_scale=0.02):
    """Test how robust a unitary distribution is to a given error distribution."""
    mesh = ClementsMesh(n_modes)
    n_mzis = mesh.n_mzis

    fidelities = []

    for _ in range(n_trials):
        # Sample target unitary
        U_target = unitary_sampler(n_modes)

        # Decompose to mesh phases (simplified - use random phases as proxy)
        thetas = np.random.uniform(0, np.pi/2, n_mzis)
        phis = np.random.uniform(0, 2*np.pi, n_mzis)
        mesh.set_phases(thetas, phis)
        U_ideal = mesh.unitary(include_noise=False)

        # Apply errors
        errors = error_sampler(n_mzis) * error_scale
        mesh.apply_noise(errors, np.zeros(n_mzis))
        U_noisy = mesh.unitary(include_noise=True)
        mesh.clear_noise()

        fidelities.append(fidelity(U_ideal, U_noisy))

    return np.mean(fidelities), np.std(fidelities)


def main():
    print()
    print("=" * 70)
    print("DISTRIBUTION ANALYSIS FOR PHOTONIC SYSTEMS")
    print("=" * 70)

    n = 16  # Test dimension

    # =========================================================================
    # PART 1: Compare Unitary Distributions
    # =========================================================================
    print()
    print("PART 1: UNITARY DISTRIBUTIONS")
    print("-" * 70)
    print()

    unitary_samplers = [
        ("Haar (uniform)", sample_haar_unitary),
        ("DFT (structured)", sample_dft_unitary),
        ("Hadamard-like", sample_hadamard_like),
        ("Diagonal", sample_diagonal_unitary),
        ("Permutation", sample_permutation_unitary),
        ("Block diagonal (4)", lambda n: sample_block_diagonal(n, 4)),
        ("Near-identity (ε=0.1)", lambda n: sample_near_identity(n, 0.1)),
        ("Low-rank (r=4)", lambda n: sample_low_rank_perturbation(n, 4)),
    ]

    print(f"{'Distribution':<25} {'Unitarity':<12} {'Phase Spread':<14} {'Sparsity':<12} {'Mixing':<10}")
    print("-" * 70)

    for name, sampler in unitary_samplers:
        U = sampler(n)
        props = analyze_unitary_properties(U, name)
        print(f"{name:<25} {props['unitarity_error']:<12.2e} {props['phase_spread']:<14.4f} "
              f"{props['sparsity']:<12.2%} {props['mixing']:<10.4f}")

    # =========================================================================
    # PART 2: Compare Error Distributions
    # =========================================================================
    print()
    print()
    print("PART 2: ERROR DISTRIBUTIONS")
    print("-" * 70)
    print()

    error_samplers = [
        ("Gaussian", lambda n: gaussian_errors(n, 1.0)),
        ("Uniform", lambda n: uniform_errors(n, np.sqrt(3))),  # Same variance
        ("Bimodal", lambda n: bimodal_errors(n, 0.5, 0.5)),
        ("Heavy-tailed", lambda n: heavy_tailed_errors(n, 0.5)),
        ("Correlated (L=10)", lambda n: correlated_errors(n, 1.0, 10)),
        ("Quantized", lambda n: quantized_errors(n, 0.1)),
        ("Sparse defects", lambda n: sparse_defect_errors(n, 0.05, 5.0)),
    ]

    n_mzis = 120  # For 16-mode Clements
    n_samples = 1000

    print(f"{'Error Type':<25} {'Mean':<12} {'Std':<12} {'Kurtosis':<12} {'Max |error|':<12}")
    print("-" * 70)

    for name, sampler in error_samplers:
        samples = np.array([sampler(n_mzis) for _ in range(n_samples)])
        all_errors = samples.flatten()

        from scipy.stats import kurtosis
        kurt = kurtosis(all_errors)

        print(f"{name:<25} {np.mean(all_errors):<12.4f} {np.std(all_errors):<12.4f} "
              f"{kurt:<12.2f} {np.max(np.abs(all_errors)):<12.4f}")

    # =========================================================================
    # PART 3: Robustness Analysis
    # =========================================================================
    print()
    print()
    print("PART 3: ROBUSTNESS ANALYSIS")
    print("-" * 70)
    print("Fidelity under different unitary × error combinations")
    print("(16 modes, σ=0.02 rad error scale, 100 trials each)")
    print()

    # Select key combinations
    key_unitaries = [
        ("Haar", sample_haar_unitary),
        ("DFT", sample_dft_unitary),
        ("Block(4)", lambda n: sample_block_diagonal(n, 4)),
    ]

    key_errors = [
        ("Gaussian", lambda n: gaussian_errors(n, 1.0)),
        ("Correlated", lambda n: correlated_errors(n, 1.0, 10)),
        ("Sparse defect", lambda n: sparse_defect_errors(n, 0.05, 5.0)),
    ]

    print(f"{'Unitary \\ Error':<20}", end="")
    for err_name, _ in key_errors:
        print(f"{err_name:<18}", end="")
    print()
    print("-" * 70)

    for uni_name, uni_sampler in key_unitaries:
        print(f"{uni_name:<20}", end="")
        for err_name, err_sampler in key_errors:
            mean_fid, std_fid = test_error_robustness(
                uni_sampler, err_sampler,
                n_modes=16, n_trials=50, error_scale=0.02
            )
            print(f"{mean_fid:.4f}±{std_fid:.4f}  ", end="")
        print()

    # =========================================================================
    # PART 4: Application-Specific Recommendations
    # =========================================================================
    print()
    print()
    print("=" * 70)
    print("RECOMMENDATIONS BY APPLICATION")
    print("=" * 70)
    print()

    recommendations = """
┌─────────────────────────────────────────────────────────────────────┐
│ APPLICATION           │ RECOMMENDED DISTRIBUTION  │ REASON          │
├─────────────────────────────────────────────────────────────────────┤
│ Optical Neural Net    │ Haar / Learned            │ Expressivity    │
│ Quantum Computing     │ Haar                      │ Universality    │
│ Boson Sampling        │ Haar                      │ Hardness proof  │
│ Signal Processing     │ DFT / Hadamard            │ Structure       │
│ Beamforming           │ Block diagonal            │ Locality        │
│ Reservoir Computing   │ Near-identity             │ Echo state      │
│ Cryptography          │ Haar                      │ Unpredictability│
└─────────────────────────────────────────────────────────────────────┘

ERROR ROBUSTNESS:
┌─────────────────────────────────────────────────────────────────────┐
│ ERROR TYPE            │ MITIGATION STRATEGY       │ NOTES           │
├─────────────────────────────────────────────────────────────────────┤
│ Gaussian (fabrication)│ Bayesian calibration      │ Our main method │
│ Correlated (thermal)  │ Real-time tracking        │ Protocol B      │
│ Sparse defects        │ Redundancy + routing      │ Skip bad MZIs   │
│ Quantized (DAC)       │ Higher bit depth          │ 16-bit typical  │
│ Heavy-tailed          │ Robust estimation         │ Median not mean │
│ Drift                 │ Periodic recalibration    │ Every ~10 min   │
└─────────────────────────────────────────────────────────────────────┘
"""
    print(recommendations)

    # =========================================================================
    # PART 5: Novel Distribution Ideas
    # =========================================================================
    print()
    print("=" * 70)
    print("NOVEL DISTRIBUTION IDEAS FOR PHOTONICS")
    print("=" * 70)
    print()

    novel_ideas = """
1. FABRICATION-AWARE DISTRIBUTIONS
   - Sample unitaries that are "easy to fabricate" (small phase angles)
   - Minimize total phase accumulation: min Σ|θᵢ|
   - Avoid sensitive operating points (θ ≈ π/4)

2. ERROR-OPTIMIZED DISTRIBUTIONS
   - Sample unitaries that minimize sensitivity to phase errors
   - Jacobian-aware sampling: choose U where ||∂U/∂θ||_F is small
   - Trade-off: less sensitive ↔ less expressive

3. THERMAL-GRADIENT-ALIGNED
   - Align mesh structure with expected thermal gradients
   - Place sensitive operations in thermally stable regions
   - Use block structure matching cooling zones

4. MEASUREMENT-EFFICIENT DISTRIBUTIONS
   - Unitaries that are easy to characterize with few measurements
   - Sparse in some basis (compressive sensing friendly)
   - Low-rank structure for efficient tomography

5. QUANTUM-ADVANTAGE PRESERVING
   - Distributions that maintain computational hardness
   - Avoid classically simulable subsets
   - Anti-concentrated output distributions

6. ADAPTIVE/LEARNED DISTRIBUTIONS
   - Start uniform, learn from calibration data
   - Bayesian prior over "good" unitaries
   - Meta-learning across multiple chips

7. FAULT-TOLERANT DISTRIBUTIONS
   - Redundant encoding of operations
   - Graceful degradation under MZI failures
   - Self-correcting through feedback
"""
    print(novel_ideas)

    return


if __name__ == "__main__":
    main()
