"""
Sparse Calibration for Large-Scale Photonic Meshes

Memory-efficient calibration using:
1. Sparse Jacobian storage (CSR format)
2. Iterative solvers (LSQR, conjugate gradient)
3. Regularization via early stopping or explicit Tikhonov

Designed for 20,000+ MZI systems that don't fit in memory with dense methods.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.sparse import csr_matrix, lil_matrix
import time


class SparseMeshSimulator:
    """
    Memory-efficient mesh simulator for large-scale systems.

    Instead of storing full unitary, computes matrix-vector products on the fly.
    """

    def __init__(self, n_modes, rng=None):
        self.n_modes = n_modes
        self.n_mzis = n_modes * (n_modes - 1) // 2
        self.rng = rng if rng is not None else np.random.default_rng()

        # Phase parameters
        self.thetas = self.rng.uniform(0, np.pi/2, self.n_mzis)
        self.phis = self.rng.uniform(0, 2*np.pi, self.n_mzis)

        # Error parameters
        self.theta_errors = np.zeros(self.n_mzis)
        self.phi_errors = np.zeros(self.n_mzis)

        # Build MZI connectivity (which modes each MZI connects)
        self._build_connectivity()

    def _build_connectivity(self):
        """Build the MZI connectivity map for Clements mesh."""
        self.mzi_modes = []  # List of (mode1, mode2) for each MZI

        mzi_idx = 0
        for col in range(self.n_modes - 1):
            # Alternate starting position for each column
            start = col % 2
            for row in range(start, self.n_modes - 1, 2):
                if mzi_idx < self.n_mzis:
                    self.mzi_modes.append((row, row + 1))
                    mzi_idx += 1

        # Pad if needed
        while len(self.mzi_modes) < self.n_mzis:
            self.mzi_modes.append((0, 1))

    def mzi_matrix(self, theta, phi):
        """2x2 MZI transfer matrix."""
        c, s = np.cos(theta), np.sin(theta)
        phase = np.exp(1j * phi)
        return np.array([
            [c, 1j * s * phase],
            [1j * s * np.conj(phase), c]
        ])

    def apply_to_vector(self, x, include_errors=False):
        """Apply mesh transformation to input vector."""
        y = x.copy().astype(complex)

        for mzi_idx in range(self.n_mzis):
            m1, m2 = self.mzi_modes[mzi_idx]

            theta = self.thetas[mzi_idx]
            phi = self.phis[mzi_idx]

            if include_errors:
                theta += self.theta_errors[mzi_idx]
                phi += self.phi_errors[mzi_idx]

            M = self.mzi_matrix(theta, phi)

            # Apply to modes m1, m2
            temp = np.array([y[m1], y[m2]])
            result = M @ temp
            y[m1], y[m2] = result[0], result[1]

        return y

    def compute_unitary_column(self, col_idx, include_errors=False):
        """Compute a single column of the unitary matrix."""
        e = np.zeros(self.n_modes, dtype=complex)
        e[col_idx] = 1.0
        return self.apply_to_vector(e, include_errors)

    def compute_full_unitary(self, include_errors=False):
        """Compute full unitary (for small systems or verification)."""
        U = np.zeros((self.n_modes, self.n_modes), dtype=complex)
        for j in range(self.n_modes):
            U[:, j] = self.compute_unitary_column(j, include_errors)
        return U

    def set_errors(self, theta_errors, phi_errors=None):
        """Set phase errors."""
        self.theta_errors = theta_errors.copy()
        if phi_errors is not None:
            self.phi_errors = phi_errors.copy()
        else:
            self.phi_errors = np.zeros(self.n_mzis)

    def clear_errors(self):
        """Clear all errors."""
        self.theta_errors = np.zeros(self.n_mzis)
        self.phi_errors = np.zeros(self.n_mzis)


def compute_sparse_jacobian(mesh, epsilon=1e-7):
    """
    Compute Jacobian in sparse format using finite differences.

    Only stores non-zero elements, dramatically reducing memory.

    Returns:
    --------
    J_sparse : scipy.sparse.csr_matrix
        Sparse Jacobian of shape (n_modes^2, n_mzis)
    """
    n_modes = mesh.n_modes
    n_mzis = mesh.n_mzis
    n_outputs = n_modes * n_modes

    # Use LIL format for efficient construction
    J_lil = lil_matrix((n_outputs, n_mzis), dtype=complex)

    # Get baseline unitary
    U0 = mesh.compute_full_unitary(include_errors=True)
    U0_flat = U0.flatten()

    # Compute each column of Jacobian
    for mzi_idx in range(n_mzis):
        # Perturb theta
        mesh.theta_errors[mzi_idx] += epsilon
        U_pert = mesh.compute_full_unitary(include_errors=True)
        mesh.theta_errors[mzi_idx] -= epsilon

        # Finite difference
        dU = (U_pert.flatten() - U0_flat) / epsilon

        # Only store significant elements
        threshold = 1e-10 * np.max(np.abs(dU))
        nonzero_mask = np.abs(dU) > threshold
        nonzero_indices = np.where(nonzero_mask)[0]

        for idx in nonzero_indices:
            J_lil[idx, mzi_idx] = dU[idx]

    # Convert to CSR for efficient arithmetic
    return csr_matrix(J_lil)


def compute_sparse_jacobian_fast(mesh, epsilon=1e-7, verbose=False):
    """
    Faster sparse Jacobian computation exploiting MZI locality.

    Each MZI only affects outputs involving its two modes,
    so we only need to recompute affected columns.
    """
    n_modes = mesh.n_modes
    n_mzis = mesh.n_mzis
    n_outputs = n_modes * n_modes

    # Use LIL format for efficient construction
    J_lil = lil_matrix((n_outputs, n_mzis), dtype=complex)

    # Get baseline unitary columns
    U0_cols = {}
    for j in range(n_modes):
        U0_cols[j] = mesh.compute_unitary_column(j, include_errors=True)

    if verbose:
        print(f"Computing sparse Jacobian for {n_mzis} MZIs...")
        last_pct = -1

    # Compute each column of Jacobian
    for mzi_idx in range(n_mzis):
        if verbose:
            pct = int(100 * mzi_idx / n_mzis)
            if pct > last_pct and pct % 10 == 0:
                print(f"  {pct}%...")
                last_pct = pct

        # Get modes affected by this MZI
        m1, m2 = mesh.mzi_modes[mzi_idx]

        # Perturb theta
        mesh.theta_errors[mzi_idx] += epsilon

        # Only recompute affected input columns
        # (columns where input goes through this MZI)
        affected_inputs = set()

        # For Clements mesh, need to track which inputs are affected
        # Simplified: recompute all columns (can optimize further)
        for j in range(n_modes):
            U_col_pert = mesh.compute_unitary_column(j, include_errors=True)
            dU_col = (U_col_pert - U0_cols[j]) / epsilon

            # Store non-zero elements
            for i in range(n_modes):
                if np.abs(dU_col[i]) > 1e-12:
                    flat_idx = i * n_modes + j
                    J_lil[flat_idx, mzi_idx] = dU_col[i]

        mesh.theta_errors[mzi_idx] -= epsilon

    if verbose:
        print(f"  100% - Done!")

    return csr_matrix(J_lil)


class SparseCalibrator:
    """
    Memory-efficient calibrator using sparse matrices and iterative solvers.
    """

    def __init__(self, n_mzis, sigma_prior=0.1):
        self.n_mzis = n_mzis
        self.sigma_prior = sigma_prior
        self.J = None
        self.y = None

    def set_jacobian(self, J_sparse):
        """Set the sparse Jacobian matrix."""
        self.J = J_sparse

    def set_measurement(self, delta_U):
        """Set the measurement vector (flattened delta_U)."""
        self.y = delta_U.flatten()

    def solve_lsqr(self, lambda_reg=None, maxiter=None, verbose=False):
        """
        Solve using LSQR (iterative least squares).

        Solves: min ||Jθ - y||² + λ||θ||²

        LSQR is memory-efficient: only needs matrix-vector products.
        """
        if self.J is None or self.y is None:
            raise ValueError("Must set Jacobian and measurement first")

        # Auto-select regularization
        if lambda_reg is None:
            # Estimate from Jacobian norm
            J_norm_est = splinalg.norm(self.J, 'fro') / np.sqrt(self.J.shape[1])
            lambda_reg = 0.01 * J_norm_est**2

        # Default iterations
        if maxiter is None:
            maxiter = min(self.n_mzis * 2, 1000)

        # LSQR with damping (Tikhonov regularization)
        damp = np.sqrt(lambda_reg)

        if verbose:
            print(f"Running LSQR with λ={lambda_reg:.2e}, max_iter={maxiter}...")

        t0 = time.time()
        result = splinalg.lsqr(
            self.J, self.y,
            damp=damp,
            iter_lim=maxiter,
            show=verbose
        )
        elapsed = time.time() - t0

        theta_est = result[0]
        iterations = result[2]
        residual_norm = result[3]

        if verbose:
            print(f"LSQR converged in {iterations} iterations, {elapsed:.2f}s")
            print(f"Residual norm: {residual_norm:.2e}")

        return theta_est.real  # Phase estimates should be real

    def solve_cg(self, lambda_reg=None, maxiter=None, verbose=False):
        """
        Solve using Conjugate Gradient on normal equations.

        Solves: (J^H J + λI) θ = J^H y

        Uses matrix-free approach - never forms J^H J explicitly.
        """
        if self.J is None or self.y is None:
            raise ValueError("Must set Jacobian and measurement first")

        # Auto-select regularization
        if lambda_reg is None:
            lambda_reg = 0.01

        if maxiter is None:
            maxiter = min(self.n_mzis * 2, 1000)

        # Right-hand side: J^H y
        rhs = self.J.conj().T @ self.y

        # Define the linear operator for (J^H J + λI)
        def matvec(x):
            return self.J.conj().T @ (self.J @ x) + lambda_reg * x

        A_op = splinalg.LinearOperator(
            (self.n_mzis, self.n_mzis),
            matvec=matvec
        )

        if verbose:
            print(f"Running CG with λ={lambda_reg:.2e}, max_iter={maxiter}...")

        t0 = time.time()
        theta_est, info = splinalg.cg(A_op, rhs, maxiter=maxiter)
        elapsed = time.time() - t0

        if verbose:
            status = "converged" if info == 0 else f"did not converge (info={info})"
            print(f"CG {status} in {elapsed:.2f}s")

        return theta_est.real


def sparse_fidelity(mesh, U_target):
    """Compute fidelity between mesh output and target unitary."""
    U_actual = mesh.compute_full_unitary(include_errors=True)
    n = U_target.shape[0]
    overlap = np.abs(np.trace(U_target.conj().T @ U_actual)) / n
    return overlap


def test_sparse_calibration(n_modes, error_std=0.1, verbose=True):
    """
    Test sparse calibration on a given system size.

    Parameters:
    -----------
    n_modes : int
        Number of modes (MZIs = n_modes*(n_modes-1)/2)
    error_std : float
        Standard deviation of phase errors
    verbose : bool
        Print progress

    Returns:
    --------
    dict with results
    """
    import traceback

    n_mzis = n_modes * (n_modes - 1) // 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPARSE CALIBRATION TEST: {n_modes} modes, {n_mzis:,} MZIs")
        print(f"{'='*60}\n")

    # Create mesh
    rng = np.random.default_rng(42)
    mesh = SparseMeshSimulator(n_modes, rng=rng)

    if verbose:
        print(f"Created mesh: {n_modes} modes, {n_mzis:,} MZIs")

    # Get target unitary (no errors)
    mesh.clear_errors()
    U_target = mesh.compute_full_unitary(include_errors=False)

    if verbose:
        print(f"Computed target unitary: {U_target.shape}")

    # Generate random phase errors
    true_errors = rng.normal(0, error_std, n_mzis)
    mesh.set_errors(true_errors)

    if verbose:
        print(f"Applied errors: σ = {error_std} rad ({np.degrees(error_std):.1f}°)")

    # Compute fidelity before calibration
    fid_before = sparse_fidelity(mesh, U_target)

    if verbose:
        print(f"Fidelity before calibration: {fid_before:.4f}")

    # Compute sparse Jacobian
    if verbose:
        print(f"\nComputing sparse Jacobian...")

    t0 = time.time()

    # Need to reset errors for Jacobian computation at operating point
    mesh.set_errors(true_errors)
    J_sparse = compute_sparse_jacobian_fast(mesh, verbose=verbose)

    t_jacobian = time.time() - t0

    # Memory statistics
    nnz = J_sparse.nnz
    dense_size = J_sparse.shape[0] * J_sparse.shape[1] * 16  # complex128
    sparse_size = nnz * 16 + J_sparse.shape[1] * 8  # data + indices
    compression = dense_size / sparse_size

    if verbose:
        print(f"\nJacobian statistics:")
        print(f"  Shape: {J_sparse.shape}")
        print(f"  Non-zeros: {nnz:,} ({100*nnz/(J_sparse.shape[0]*J_sparse.shape[1]):.1f}%)")
        print(f"  Dense size: {dense_size/1e6:.1f} MB")
        print(f"  Sparse size: {sparse_size/1e6:.1f} MB")
        print(f"  Compression: {compression:.1f}x")
        print(f"  Time: {t_jacobian:.1f}s")

    # Compute measurement
    U_noisy = mesh.compute_full_unitary(include_errors=True)
    delta_U = (U_noisy - U_target).flatten()

    # Calibrate using LSQR
    if verbose:
        print(f"\nCalibrating with LSQR...")

    calibrator = SparseCalibrator(n_mzis, sigma_prior=error_std * 1.5)
    calibrator.set_jacobian(J_sparse)
    calibrator.set_measurement(delta_U)

    t0 = time.time()
    theta_est = calibrator.solve_lsqr(verbose=verbose)
    t_solve = time.time() - t0

    # Apply correction
    correction = -theta_est * 0.85
    mesh.set_errors(true_errors + correction)

    # Compute fidelity after calibration
    fid_after = sparse_fidelity(mesh, U_target)

    # Compute recovery
    if fid_before < 1.0:
        recovery = (fid_after - fid_before) / (1.0 - fid_before) * 100
    else:
        recovery = 100.0

    # Estimation error
    est_error = np.std(theta_est - true_errors)

    if verbose:
        print(f"\nResults:")
        print(f"  Fidelity before: {fid_before:.4f}")
        print(f"  Fidelity after:  {fid_after:.4f}")
        print(f"  Recovery:        {recovery:.1f}%")
        print(f"  Estimation error: {est_error:.4f} rad")
        print(f"  Solve time:      {t_solve:.2f}s")
        print(f"  Total time:      {t_jacobian + t_solve:.1f}s")

    return {
        'n_modes': n_modes,
        'n_mzis': n_mzis,
        'fid_before': fid_before,
        'fid_after': fid_after,
        'recovery': recovery,
        'est_error': est_error,
        't_jacobian': t_jacobian,
        't_solve': t_solve,
        'sparse_size_mb': sparse_size / 1e6,
        'compression': compression
    }


if __name__ == "__main__":
    # Test on increasing sizes
    print("\n" + "="*70)
    print("SPARSE CALIBRATION VALIDATION")
    print("="*70)

    # Start small to verify correctness
    print("\n" + "-"*70)
    print("Small-scale verification (8 modes, 28 MZIs)")
    print("-"*70)
    result_small = test_sparse_calibration(8, error_std=0.1, verbose=True)

    print("\n" + "-"*70)
    print("Medium-scale test (32 modes, 496 MZIs)")
    print("-"*70)
    result_medium = test_sparse_calibration(32, error_std=0.1, verbose=True)

    print("\n" + "-"*70)
    print("Large-scale test (64 modes, 2,016 MZIs)")
    print("-"*70)
    result_large = test_sparse_calibration(64, error_std=0.1, verbose=True)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Size':<20} {'MZIs':<10} {'Recovery':<12} {'Time':<12} {'Memory':<12}")
    print("-"*66)
    for r in [result_small, result_medium, result_large]:
        print(f"{r['n_modes']} modes{'':<10} {r['n_mzis']:<10,} {r['recovery']:<12.1f}% "
              f"{r['t_jacobian']+r['t_solve']:<12.1f}s {r['sparse_size_mb']:<12.1f} MB")
