"""
Parallel sparse calibration using multiprocessing.
Uses all CPU cores to compute Jacobian columns in parallel.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import linalg as splinalg
import multiprocessing as mp
import time

# Global variables for worker processes
_mesh_data = None

def init_worker(thetas, phis, mzi_modes, n_modes):
    """Initialize worker with mesh data."""
    global _mesh_data
    _mesh_data = {
        'thetas': thetas,
        'phis': phis,
        'mzi_modes': mzi_modes,
        'n_modes': n_modes
    }


def compute_unitary_static(thetas, phis, mzi_modes, n_modes):
    """Compute unitary from phases."""
    U = np.eye(n_modes, dtype=complex)
    for mzi_idx in range(len(mzi_modes)):
        m1, m2 = mzi_modes[mzi_idx]
        c, s = np.cos(thetas[mzi_idx]), np.sin(thetas[mzi_idx])
        phase = np.exp(1j * phis[mzi_idx])

        U_new = U.copy()
        U_new[m1, :] = c * U[m1, :] + 1j * s * phase * U[m2, :]
        U_new[m2, :] = 1j * s * np.conj(phase) * U[m1, :] + c * U[m2, :]
        U = U_new
    return U


def compute_jacobian_column(args):
    """Compute single column of Jacobian."""
    mzi_idx, theta_errors = args
    global _mesh_data

    thetas = _mesh_data['thetas']
    phis = _mesh_data['phis']
    mzi_modes = _mesh_data['mzi_modes']
    n_modes = _mesh_data['n_modes']

    eps = 1e-7
    thetas_with_err = thetas + theta_errors

    # Baseline
    U0 = compute_unitary_static(thetas_with_err, phis, mzi_modes, n_modes)

    # Perturbed
    thetas_pert = thetas_with_err.copy()
    thetas_pert[mzi_idx] += eps
    U_pert = compute_unitary_static(thetas_pert, phis, mzi_modes, n_modes)

    # Derivative column
    dU = (U_pert - U0).flatten() / eps

    # Return sparse representation
    nonzero_mask = np.abs(dU) > 1e-12
    indices = np.where(nonzero_mask)[0]
    values = dU[nonzero_mask]

    return mzi_idx, indices, values


class ParallelMesh:
    """Mesh simulator for parallel Jacobian computation."""

    def __init__(self, n_modes, rng_seed=42):
        self.n_modes = n_modes
        self.n_mzis = n_modes * (n_modes - 1) // 2
        rng = np.random.default_rng(rng_seed)

        self.thetas = rng.uniform(0, np.pi/2, self.n_mzis)
        self.phis = rng.uniform(0, 2*np.pi, self.n_mzis)

        # Build MZI structure
        self.mzi_modes = []
        mzi_idx = 0
        for col in range(n_modes - 1):
            start = col % 2
            for row in range(start, n_modes - 1, 2):
                if mzi_idx < self.n_mzis:
                    self.mzi_modes.append((row, row + 1))
                    mzi_idx += 1
        while len(self.mzi_modes) < self.n_mzis:
            self.mzi_modes.append((0, 1))
        self.mzi_modes = np.array(self.mzi_modes)

    def compute_unitary(self, theta_errors=None):
        """Compute unitary."""
        thetas = self.thetas + (theta_errors if theta_errors is not None else 0)
        return compute_unitary_static(thetas, self.phis, self.mzi_modes, self.n_modes)


def compute_jacobian_parallel(mesh, theta_errors, n_workers=None):
    """Compute Jacobian using parallel workers."""
    if n_workers is None:
        n_workers = mp.cpu_count()

    n_mzis = mesh.n_mzis
    n_modes = mesh.n_modes
    n_outputs = n_modes * n_modes

    # Prepare arguments
    args_list = [(k, theta_errors) for k in range(n_mzis)]

    # Parallel computation with initializer
    with mp.Pool(
        n_workers,
        initializer=init_worker,
        initargs=(mesh.thetas, mesh.phis, mesh.mzi_modes, mesh.n_modes)
    ) as pool:
        results = pool.map(compute_jacobian_column, args_list)

    # Assemble sparse matrix
    data = []
    row_indices = []
    col_indices = []

    for mzi_idx, indices, values in results:
        data.extend(values)
        row_indices.extend(indices)
        col_indices.extend([mzi_idx] * len(indices))

    J_sparse = csr_matrix(
        (data, (row_indices, col_indices)),
        shape=(n_outputs, n_mzis),
        dtype=complex
    )

    return J_sparse


def test_parallel_calibration(n_modes, error_std=0.05, verbose=True):
    """Test parallel sparse calibration."""
    n_mzis = n_modes * (n_modes - 1) // 2
    n_cores = mp.cpu_count()

    if verbose:
        print(f"\n{'='*60}")
        print(f"PARALLEL CALIBRATION: {n_modes} modes, {n_mzis:,} MZIs ({n_cores} cores)")
        print(f"{'='*60}")

    mesh = ParallelMesh(n_modes)
    rng = np.random.default_rng(42)

    # Target unitary
    U_target = mesh.compute_unitary()

    # Errors
    true_errors = rng.normal(0, error_std, n_mzis)
    U_noisy = mesh.compute_unitary(true_errors)

    fid_before = np.abs(np.trace(U_target.conj().T @ U_noisy)) / n_modes
    if verbose:
        print(f"Fidelity before: {fid_before:.4f}")

    # Parallel Jacobian computation
    if verbose:
        print(f"\nComputing Jacobian with {n_cores} cores...")
    t0 = time.time()
    J_sparse = compute_jacobian_parallel(mesh, true_errors)
    t_jac = time.time() - t0

    nnz = J_sparse.nnz
    mem_mb = nnz * 16 / 1e6
    if verbose:
        print(f"Jacobian: {J_sparse.shape}, {nnz:,} non-zeros")
        print(f"Memory: {mem_mb:.1f} MB, Time: {t_jac:.1f}s")

    # Solve
    if verbose:
        print(f"\nSolving with LSQR...")
    delta_U = (U_noisy - U_target).flatten()

    t0 = time.time()
    result = splinalg.lsqr(J_sparse, delta_U, damp=0.1, iter_lim=500, show=False)
    t_solve = time.time() - t0

    theta_est = result[0].real
    correction = -theta_est * 0.85

    # Calibrated result
    U_calib = mesh.compute_unitary(true_errors + correction)
    fid_after = np.abs(np.trace(U_target.conj().T @ U_calib)) / n_modes

    recovery = (fid_after - fid_before) / (1 - fid_before) * 100 if fid_before < 1 else 100

    if verbose:
        print(f"Solve time: {t_solve:.2f}s")
        print(f"\nResults:")
        print(f"  Fidelity: {fid_before:.4f} → {fid_after:.4f}")
        print(f"  Recovery: {recovery:.1f}%")
        print(f"  Total time: {t_jac + t_solve:.1f}s")

    return {
        'n_modes': n_modes,
        'n_mzis': n_mzis,
        'fid_before': fid_before,
        'fid_after': fid_after,
        'recovery': recovery,
        't_jacobian': t_jac,
        't_solve': t_solve,
        't_total': t_jac + t_solve,
        'mem_mb': mem_mb
    }


if __name__ == "__main__":
    print(f"\nCPU cores available: {mp.cpu_count()}")

    # Test scaling with parallel computation
    results = []
    for n_modes in [32, 64, 100, 150]:
        r = test_parallel_calibration(n_modes, error_std=0.05)
        results.append(r)

    print("\n" + "="*70)
    print("PARALLEL SCALING SUMMARY")
    print("="*70)
    print(f"\n{'MZIs':<12} {'Recovery':<12} {'Jacobian':<12} {'Solve':<12} {'Total':<12}")
    print("-"*60)
    for r in results:
        print(f"{r['n_mzis']:<12,} {r['recovery']:<12.1f}% {r['t_jacobian']:<12.1f}s {r['t_solve']:<12.2f}s {r['t_total']:<12.1f}s")
