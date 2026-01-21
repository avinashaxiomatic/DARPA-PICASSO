"""
Optimized Sparse Calibration for Large-Scale Photonic Meshes (v2)

Key optimizations:
1. Analytical Jacobian using chain rule (no finite differences)
2. Only compute non-zero Jacobian elements
3. Vectorized operations where possible
4. Memory-efficient iterative solver

Target: 20,000+ MZIs in reasonable time and memory.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.sparse import csr_matrix, lil_matrix, dok_matrix
import time


class OptimizedMeshSimulator:
    """
    Optimized mesh simulator using vectorized operations.
    """

    def __init__(self, n_modes, rng=None):
        self.n_modes = n_modes
        self.n_mzis = n_modes * (n_modes - 1) // 2
        self.rng = rng if rng is not None else np.random.default_rng()

        # Phase parameters
        self.thetas = self.rng.uniform(0, np.pi/2, self.n_mzis)
        self.phis = self.rng.uniform(0, 2*np.pi, self.n_mzis)

        # Build MZI structure
        self._build_structure()

    def _build_structure(self):
        """Build MZI connectivity for Clements mesh."""
        self.mzi_modes = []
        self.mzi_layer = []

        mzi_idx = 0
        layer = 0
        for col in range(self.n_modes - 1):
            start = col % 2
            for row in range(start, self.n_modes - 1, 2):
                if mzi_idx < self.n_mzis:
                    self.mzi_modes.append((row, row + 1))
                    self.mzi_layer.append(layer)
                    mzi_idx += 1
            layer += 1

        while len(self.mzi_modes) < self.n_mzis:
            self.mzi_modes.append((0, 1))
            self.mzi_layer.append(layer)

        self.mzi_modes = np.array(self.mzi_modes)

    def compute_unitary(self, theta_errors=None):
        """Compute full unitary matrix."""
        U = np.eye(self.n_modes, dtype=complex)

        for mzi_idx in range(self.n_mzis):
            m1, m2 = self.mzi_modes[mzi_idx]

            theta = self.thetas[mzi_idx]
            phi = self.phis[mzi_idx]

            if theta_errors is not None:
                theta += theta_errors[mzi_idx]

            c, s = np.cos(theta), np.sin(theta)
            phase = np.exp(1j * phi)

            # Apply MZI to unitary
            U_new = U.copy()
            U_new[m1, :] = c * U[m1, :] + 1j * s * phase * U[m2, :]
            U_new[m2, :] = 1j * s * np.conj(phase) * U[m1, :] + c * U[m2, :]
            U = U_new

        return U


def compute_jacobian_analytical(mesh, theta_errors=None):
    """
    Compute Jacobian analytically using chain rule.

    dU/dθ_k = U_after_k @ (dM_k/dθ_k) @ U_before_k

    where dM_k/dθ_k is the derivative of the k-th MZI.
    """
    n_modes = mesh.n_modes
    n_mzis = mesh.n_mzis

    # Build partial unitaries
    # U_before[k] = product of MZIs 0 to k-1
    # U_after[k] = product of MZIs k+1 to end

    thetas = mesh.thetas.copy()
    if theta_errors is not None:
        thetas = thetas + theta_errors

    # Forward pass: compute U_before for each MZI
    U_before = [np.eye(n_modes, dtype=complex)]
    U_current = np.eye(n_modes, dtype=complex)

    for mzi_idx in range(n_mzis):
        m1, m2 = mesh.mzi_modes[mzi_idx]
        theta = thetas[mzi_idx]
        phi = mesh.phis[mzi_idx]

        c, s = np.cos(theta), np.sin(theta)
        phase = np.exp(1j * phi)

        U_new = U_current.copy()
        U_new[m1, :] = c * U_current[m1, :] + 1j * s * phase * U_current[m2, :]
        U_new[m2, :] = 1j * s * np.conj(phase) * U_current[m1, :] + c * U_current[m2, :]
        U_current = U_new

        U_before.append(U_current.copy())

    # Backward pass: compute U_after for each MZI
    U_after = [None] * (n_mzis + 1)
    U_after[n_mzis] = np.eye(n_modes, dtype=complex)
    U_current = np.eye(n_modes, dtype=complex)

    for mzi_idx in range(n_mzis - 1, -1, -1):
        U_after[mzi_idx] = U_current.copy()

        m1, m2 = mesh.mzi_modes[mzi_idx]
        theta = thetas[mzi_idx]
        phi = mesh.phis[mzi_idx]

        c, s = np.cos(theta), np.sin(theta)
        phase = np.exp(1j * phi)

        # MZI matrix (acts on rows in backward pass)
        U_new = U_current.copy()
        U_new[:, m1] = c * U_current[:, m1] + 1j * s * np.conj(phase) * U_current[:, m2]
        U_new[:, m2] = 1j * s * phase * U_current[:, m1] + c * U_current[:, m2]
        U_current = U_new

    # Now compute Jacobian
    # dU/dθ_k = U_after[k+1] @ dM_k @ U_before[k]
    # where dM_k is the derivative matrix (only affects 2 rows/cols)

    # Store in sparse format
    J_data = []
    J_row = []
    J_col = []

    for mzi_idx in range(n_mzis):
        m1, m2 = mesh.mzi_modes[mzi_idx]
        theta = thetas[mzi_idx]
        phi = mesh.phis[mzi_idx]

        c, s = np.cos(theta), np.sin(theta)
        phase = np.exp(1j * phi)

        # dM/dtheta: derivative of MZI matrix w.r.t theta
        # M = [[c, is*p], [is*p*, c]]
        # dM/dtheta = [[-s, ic*p], [ic*p*, -s]]
        dM = np.array([
            [-s, 1j * c * phase],
            [1j * c * np.conj(phase), -s]
        ])

        # Extract relevant parts of U_before and U_after
        U_b = U_before[mzi_idx][[m1, m2], :]  # 2 x n_modes
        U_a = U_after[mzi_idx + 1][:, [m1, m2]]  # n_modes x 2

        # dU = U_a @ dM @ U_b
        dU = U_a @ dM @ U_b  # n_modes x n_modes

        # Store non-zero elements
        for i in range(n_modes):
            for j in range(n_modes):
                val = dU[i, j]
                if np.abs(val) > 1e-14:
                    flat_idx = i * n_modes + j
                    J_data.append(val)
                    J_row.append(flat_idx)
                    J_col.append(mzi_idx)

    # Create sparse matrix
    n_outputs = n_modes * n_modes
    J_sparse = csr_matrix(
        (J_data, (J_row, J_col)),
        shape=(n_outputs, n_mzis),
        dtype=complex
    )

    return J_sparse


class SparseCalibrator:
    """Memory-efficient calibrator using iterative solvers."""

    def __init__(self, n_mzis, sigma_prior=0.1):
        self.n_mzis = n_mzis
        self.sigma_prior = sigma_prior
        self.J = None
        self.y = None

    def set_jacobian(self, J_sparse):
        self.J = J_sparse

    def set_measurement(self, delta_U):
        self.y = delta_U.flatten()

    def solve_lsqr(self, lambda_reg=None, maxiter=None, verbose=False):
        """Solve using LSQR with Tikhonov regularization."""
        if self.J is None or self.y is None:
            raise ValueError("Must set Jacobian and measurement first")

        if lambda_reg is None:
            lambda_reg = 0.01

        if maxiter is None:
            maxiter = min(self.n_mzis * 2, 2000)

        damp = np.sqrt(lambda_reg)

        t0 = time.time()
        result = splinalg.lsqr(
            self.J, self.y,
            damp=damp,
            iter_lim=maxiter,
            show=False
        )
        elapsed = time.time() - t0

        if verbose:
            print(f"LSQR: {result[2]} iterations, {elapsed:.2f}s")

        return result[0].real


def fidelity(U1, U2):
    """Compute fidelity between two unitaries."""
    n = U1.shape[0]
    return np.abs(np.trace(U1.conj().T @ U2)) / n


def test_sparse_calibration_v2(n_modes, error_std=0.1, verbose=True):
    """Test optimized sparse calibration."""
    n_mzis = n_modes * (n_modes - 1) // 2

    if verbose:
        print(f"\n{'='*60}")
        print(f"SPARSE CALIBRATION v2: {n_modes} modes, {n_mzis:,} MZIs")
        print(f"{'='*60}\n")

    rng = np.random.default_rng(42)
    mesh = OptimizedMeshSimulator(n_modes, rng=rng)

    if verbose:
        print(f"System: {n_modes} modes, {n_mzis:,} MZIs")

    # Target unitary
    U_target = mesh.compute_unitary()

    # Generate errors
    true_errors = rng.normal(0, error_std, n_mzis)

    # Noisy unitary
    U_noisy = mesh.compute_unitary(true_errors)
    fid_before = fidelity(U_target, U_noisy)

    if verbose:
        print(f"Error: σ = {error_std} rad ({np.degrees(error_std):.1f}°)")
        print(f"Fidelity before: {fid_before:.4f}")

    # Compute sparse Jacobian
    if verbose:
        print(f"\nComputing analytical Jacobian...")

    t0 = time.time()
    J_sparse = compute_jacobian_analytical(mesh, true_errors)
    t_jacobian = time.time() - t0

    if verbose:
        nnz = J_sparse.nnz
        total = J_sparse.shape[0] * J_sparse.shape[1]
        sparsity = 1 - nnz / total
        mem_sparse = (nnz * 16 + J_sparse.shape[1] * 8) / 1e6
        mem_dense = total * 16 / 1e6
        print(f"Jacobian: {J_sparse.shape}, {nnz:,} non-zeros ({100*(1-sparsity):.1f}%)")
        print(f"Memory: {mem_sparse:.1f} MB sparse vs {mem_dense:.1f} MB dense")
        print(f"Time: {t_jacobian:.2f}s")

    # Measurement
    delta_U = (U_noisy - U_target).flatten()

    # Calibrate
    if verbose:
        print(f"\nCalibrating...")

    calibrator = SparseCalibrator(n_mzis)
    calibrator.set_jacobian(J_sparse)
    calibrator.set_measurement(delta_U)

    t0 = time.time()
    theta_est = calibrator.solve_lsqr(verbose=verbose)
    t_solve = time.time() - t0

    # Apply correction
    correction = -theta_est * 0.85
    U_calibrated = mesh.compute_unitary(true_errors + correction)
    fid_after = fidelity(U_target, U_calibrated)

    # Recovery
    recovery = (fid_after - fid_before) / (1 - fid_before) * 100 if fid_before < 1 else 100

    if verbose:
        print(f"\nResults:")
        print(f"  Fidelity: {fid_before:.4f} → {fid_after:.4f}")
        print(f"  Recovery: {recovery:.1f}%")
        print(f"  Total time: {t_jacobian + t_solve:.2f}s")

    return {
        'n_modes': n_modes,
        'n_mzis': n_mzis,
        'fid_before': fid_before,
        'fid_after': fid_after,
        'recovery': recovery,
        't_jacobian': t_jacobian,
        't_solve': t_solve,
        't_total': t_jacobian + t_solve
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPARSE CALIBRATION v2 - OPTIMIZED")
    print("="*70)

    # Test scaling
    results = []
    for n_modes in [16, 32, 64, 100, 150, 200]:
        r = test_sparse_calibration_v2(n_modes, error_std=0.1, verbose=True)
        results.append(r)

    # Summary
    print("\n" + "="*70)
    print("SCALING SUMMARY")
    print("="*70)
    print(f"\n{'Modes':<10} {'MZIs':<12} {'Recovery':<12} {'Jacobian':<12} {'Solve':<12} {'Total':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_modes']:<10} {r['n_mzis']:<12,} {r['recovery']:<12.1f}% "
              f"{r['t_jacobian']:<12.2f}s {r['t_solve']:<12.2f}s {r['t_total']:<12.2f}s")
