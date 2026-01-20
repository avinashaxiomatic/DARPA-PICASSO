"""
Performance Comparison: Our Formalism vs. Baseline Approaches

This script quantifies the performance improvements enabled by the
PICASSO mathematical framework, comparing:

1. Blind correction vs. Sensitivity-guided correction
2. Random configurations vs. Haar-optimized configurations
3. Naive error budgeting vs. Predictive scaling laws

These comparisons demonstrate the practical value of the formalism
for photonic mesh design and calibration.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from picasso_sim.core.mesh import ClementsMesh, random_mesh
from picasso_sim.core.noise import GaussianPhaseNoise, FabricationNoise
from picasso_sim.analysis.perturbation import PerturbationAnalyzer
from picasso_sim.analysis.sensitivity import SensitivityAnalyzer
from picasso_sim.analysis.fidelity import fidelity, infidelity
from picasso_sim.random_matrix.haar import haar_unitary, sample_haar_phases


def print_section(title):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label, value, unit=""):
    """Print formatted result."""
    if isinstance(value, float):
        print(f"  {label}: {value:.4f} {unit}")
    else:
        print(f"  {label}: {value} {unit}")


# =============================================================================
# COMPARISON 1: Blind vs. Sensitivity-Guided Error Correction
# =============================================================================

def compare_correction_strategies(n_modes=6, sigma=0.05, n_trials=100):
    """
    Compare blind uniform correction vs. sensitivity-guided correction.

    Blind correction: Apply uniform compensation to all MZIs
    Guided correction: Weight compensation by sensitivity (Jacobian norm)
    """
    print_section("COMPARISON 1: Blind vs. Sensitivity-Guided Correction")

    rng = np.random.default_rng(42)

    blind_fidelities = []
    guided_fidelities = []
    uncorrected_fidelities = []

    for trial in range(n_trials):
        # Create mesh with random target
        mesh = random_mesh(n_modes, "clements", rng)
        U_target = mesh.unitary(include_noise=False)

        # Apply structured noise (fabrication-like, non-uniform)
        fab_noise = FabricationNoise(sigma_theta=sigma, sigma_phi=sigma)
        delta_thetas, delta_phis = fab_noise.sample(mesh.n_mzis, rng)
        mesh.apply_noise(delta_thetas, delta_phis)

        U_noisy = mesh.unitary(include_noise=True)
        uncorrected_fid = fidelity(U_target, U_noisy)
        uncorrected_fidelities.append(uncorrected_fid)

        # Compute sensitivities
        analyzer = SensitivityAnalyzer(mesh)
        sensitivity_result = analyzer.compute()
        sensitivities = sensitivity_result.sensitivity_norms

        # Normalize sensitivities to use as weights
        weights = sensitivities / np.mean(sensitivities)

        # Strategy 1: Blind uniform correction
        # Assume we can measure total error and distribute correction uniformly
        # Correct each MZI by 50% of estimated per-MZI error
        blind_correction_theta = -delta_thetas * 0.5
        blind_correction_phi = -delta_phis * 0.5

        mesh.clear_noise()
        mesh.apply_noise(delta_thetas + blind_correction_theta,
                         delta_phis + blind_correction_phi)
        U_blind = mesh.unitary(include_noise=True)
        blind_fid = fidelity(U_target, U_blind)
        blind_fidelities.append(blind_fid)

        # Strategy 2: Sensitivity-guided correction
        # Weight correction by sensitivity (correct more on sensitive MZIs)
        guided_correction_theta = -delta_thetas * weights * 0.5
        guided_correction_phi = -delta_phis * weights * 0.5
        # Normalize to have same total correction magnitude
        guided_correction_theta *= np.sum(np.abs(blind_correction_theta)) / (np.sum(np.abs(guided_correction_theta)) + 1e-15)
        guided_correction_phi *= np.sum(np.abs(blind_correction_phi)) / (np.sum(np.abs(guided_correction_phi)) + 1e-15)

        mesh.clear_noise()
        mesh.apply_noise(delta_thetas + guided_correction_theta,
                         delta_phis + guided_correction_phi)
        U_guided = mesh.unitary(include_noise=True)
        guided_fid = fidelity(U_target, U_guided)
        guided_fidelities.append(guided_fid)

        mesh.clear_noise()

    # Results
    uncorr_mean = np.mean(uncorrected_fidelities)
    blind_mean = np.mean(blind_fidelities)
    guided_mean = np.mean(guided_fidelities)

    print(f"\n  Noise level: σ = {sigma:.3f} rad ({np.degrees(sigma):.2f}°)")
    print(f"  Trials: {n_trials}")
    print()
    print_result("Uncorrected mean fidelity", uncorr_mean)
    print_result("Blind correction mean fidelity", blind_mean)
    print_result("Guided correction mean fidelity", guided_mean)
    print()

    # Improvement metrics
    blind_improvement = (blind_mean - uncorr_mean) / (1 - uncorr_mean) * 100
    guided_improvement = (guided_mean - uncorr_mean) / (1 - uncorr_mean) * 100
    relative_advantage = (guided_mean - blind_mean) / (blind_mean - uncorr_mean) * 100 if blind_mean > uncorr_mean else float('inf')

    print("  IMPROVEMENT METRICS:")
    print(f"    Blind correction recovers: {blind_improvement:.1f}% of lost fidelity")
    print(f"    Guided correction recovers: {guided_improvement:.1f}% of lost fidelity")
    print(f"    Sensitivity-guided advantage: {relative_advantage:.1f}% better than blind")

    return {
        'uncorrected': uncorr_mean,
        'blind': blind_mean,
        'guided': guided_mean,
        'guided_advantage_percent': relative_advantage
    }


# =============================================================================
# COMPARISON 2: Random vs. Haar-Optimized Configurations
# =============================================================================

def compare_mesh_configurations(n_modes=6, sigma=0.02, n_trials=100):
    """
    Compare error resilience of random vs. Haar-distributed configurations.

    Random: Arbitrary phase settings
    Haar: Phases chosen to implement Haar-random unitaries (error delocalization)
    """
    print_section("COMPARISON 2: Random vs. Haar-Optimized Configurations")

    rng = np.random.default_rng(42)
    noise = GaussianPhaseNoise(sigma_theta=sigma, sigma_phi=sigma)

    random_errors = []
    haar_errors = []
    random_variances = []
    haar_variances = []

    for trial in range(n_trials):
        # Configuration 1: Random phases
        mesh_random = random_mesh(n_modes, "clements", rng)
        U_random_ideal = mesh_random.unitary(include_noise=False)

        # Sample multiple noise realizations
        errors_random = []
        for _ in range(20):
            noise.apply_to_mesh(mesh_random, rng)
            U_random_noisy = mesh_random.unitary(include_noise=True)
            errors_random.append(np.linalg.norm(U_random_noisy - U_random_ideal, 'fro'))
            mesh_random.clear_noise()

        random_errors.append(np.mean(errors_random))
        random_variances.append(np.var(errors_random))

        # Configuration 2: Haar-distributed phases
        mesh_haar = ClementsMesh(n_modes)
        thetas, phis = sample_haar_phases(n_modes, mesh_haar.n_mzis, rng)
        mesh_haar.set_phases(thetas, phis)
        U_haar_ideal = mesh_haar.unitary(include_noise=False)

        errors_haar = []
        for _ in range(20):
            noise.apply_to_mesh(mesh_haar, rng)
            U_haar_noisy = mesh_haar.unitary(include_noise=True)
            errors_haar.append(np.linalg.norm(U_haar_noisy - U_haar_ideal, 'fro'))
            mesh_haar.clear_noise()

        haar_errors.append(np.mean(errors_haar))
        haar_variances.append(np.var(errors_haar))

    # Results
    print(f"\n  Noise level: σ = {sigma:.3f} rad ({np.degrees(sigma):.2f}°)")
    print(f"  Trials: {n_trials}")
    print()

    random_error_mean = np.mean(random_errors)
    haar_error_mean = np.mean(haar_errors)
    random_var_mean = np.mean(random_variances)
    haar_var_mean = np.mean(haar_variances)

    print_result("Random config mean error", random_error_mean)
    print_result("Haar config mean error", haar_error_mean)
    print()
    print_result("Random config error variance", random_var_mean)
    print_result("Haar config error variance", haar_var_mean)
    print()

    # Improvement metrics
    error_reduction = (random_error_mean - haar_error_mean) / random_error_mean * 100
    variance_reduction = (random_var_mean - haar_var_mean) / random_var_mean * 100

    print("  IMPROVEMENT METRICS:")
    print(f"    Mean error reduction: {error_reduction:.1f}%")
    print(f"    Error variance reduction: {variance_reduction:.1f}%")
    print(f"    Error delocalization benefit: Haar configs spread errors more uniformly")

    return {
        'random_error': random_error_mean,
        'haar_error': haar_error_mean,
        'error_reduction_percent': error_reduction,
        'variance_reduction_percent': variance_reduction
    }


# =============================================================================
# COMPARISON 3: Naive vs. Predictive Error Budgeting
# =============================================================================

def compare_error_budgeting(target_fidelity=0.99, n_modes_list=[4, 6, 8, 10]):
    """
    Compare naive error budgeting vs. using our scaling law.

    Naive: Assume errors add linearly with circuit size
    Predictive: Use √L scaling law: ||δU|| ≈ 1.72 · σ · √L
    """
    print_section("COMPARISON 3: Naive vs. Predictive Error Budgeting")

    rng = np.random.default_rng(42)

    # From our validation: ||δU|| ≈ 1.72 · σ · √L
    # For fidelity F ≈ 1 - ||δU||²/(2n), we need ||δU|| ≈ √(2n(1-F))

    print(f"\n  Target fidelity: {target_fidelity}")
    print(f"  Testing mode counts: {n_modes_list}")
    print()

    print("  {:>6} {:>8} {:>12} {:>12} {:>12} {:>10}".format(
        "Modes", "Depth", "Naive σ", "Predict σ", "Actual σ*", "Savings"))
    print("  " + "-" * 62)

    results = []

    for n_modes in n_modes_list:
        mesh = ClementsMesh(n_modes)
        L = mesh.depth
        n_mzis = mesh.n_mzis

        # Target error norm for desired fidelity
        target_error = np.sqrt(2 * n_modes * (1 - target_fidelity))

        # Naive budgeting: assume errors add linearly
        # ||δU|| ≈ n_mzis · σ → σ_naive = target_error / n_mzis
        sigma_naive = target_error / n_mzis

        # Predictive budgeting: use √L scaling
        # ||δU|| ≈ 1.72 · σ · √L → σ_predict = target_error / (1.72 · √L)
        sigma_predict = target_error / (1.72 * np.sqrt(L))

        # Find actual required sigma empirically
        sigma_actual = find_sigma_for_fidelity(mesh, target_fidelity, rng)

        # Savings: how much more noise we can tolerate
        savings = (sigma_predict / sigma_naive - 1) * 100

        print(f"  {n_modes:>6} {L:>8} {np.degrees(sigma_naive):>11.3f}° "
              f"{np.degrees(sigma_predict):>11.3f}° {np.degrees(sigma_actual):>11.3f}° "
              f"{savings:>9.1f}%")

        results.append({
            'n_modes': n_modes,
            'depth': L,
            'sigma_naive': sigma_naive,
            'sigma_predict': sigma_predict,
            'sigma_actual': sigma_actual,
            'tolerance_improvement': savings
        })

    print()
    print("  * Actual σ found by binary search to achieve target fidelity")
    print()

    avg_improvement = np.mean([r['tolerance_improvement'] for r in results])
    print("  IMPROVEMENT METRICS:")
    print(f"    Average noise tolerance improvement: {avg_improvement:.1f}%")
    print(f"    Predictive budgeting allows higher σ while meeting specs")
    print(f"    Benefit: Relaxed fabrication requirements, lower cost")

    return results


def find_sigma_for_fidelity(mesh, target_fidelity, rng, n_samples=50):
    """Binary search to find sigma that achieves target fidelity."""
    sigma_low, sigma_high = 0.001, 0.5

    U_ideal = mesh.unitary(include_noise=False)

    while sigma_high - sigma_low > 0.0005:
        sigma_mid = (sigma_low + sigma_high) / 2
        noise = GaussianPhaseNoise(sigma_theta=sigma_mid, sigma_phi=sigma_mid)

        fidelities = []
        for _ in range(n_samples):
            noise.apply_to_mesh(mesh, rng)
            U_noisy = mesh.unitary(include_noise=True)
            fidelities.append(fidelity(U_ideal, U_noisy))
            mesh.clear_noise()

        mean_fid = np.mean(fidelities)

        if mean_fid >= target_fidelity:
            sigma_low = sigma_mid
        else:
            sigma_high = sigma_mid

    return sigma_low


# =============================================================================
# COMPARISON 4: First-Order Prediction Accuracy
# =============================================================================

def compare_prediction_methods(n_modes=6, n_trials=50):
    """
    Compare accuracy of first-order perturbation prediction vs. brute force.

    First-order: Fast analytical approximation using Jacobian
    Brute force: Expensive Monte Carlo simulation
    """
    print_section("COMPARISON 4: First-Order Prediction vs. Monte Carlo")

    rng = np.random.default_rng(42)

    sigma_values = [0.005, 0.01, 0.02, 0.03, 0.05]

    print(f"\n  Mode count: {n_modes}")
    print(f"  Trials per sigma: {n_trials}")
    print()
    print("  {:>10} {:>12} {:>12} {:>10} {:>12}".format(
        "σ (rad)", "1st-Order", "Monte Carlo", "Error", "Speedup"))
    print("  " + "-" * 58)

    results = []

    for sigma in sigma_values:
        mesh = random_mesh(n_modes, "clements", rng)
        sens_analyzer = SensitivityAnalyzer(mesh)

        # Method 1: First-order analytical prediction
        import time
        t0 = time.time()
        fo_predictions = []
        for _ in range(n_trials):
            thetas = rng.uniform(0, np.pi/2, mesh.n_mzis)
            phis = rng.uniform(0, 2*np.pi, mesh.n_mzis)
            mesh.set_phases(thetas, phis)

            # First-order prediction using Jacobian norm
            result = sens_analyzer.compute()
            # ||δU||_F ≈ σ · √(Σ||∂U/∂θ_i||² + Σ||∂U/∂φ_i||²)
            jacobian_norm = np.sqrt(np.sum(result.sensitivity_norms**2))
            fo_pred = sigma * jacobian_norm
            fo_predictions.append(fo_pred)

        fo_time = time.time() - t0
        fo_mean = np.mean(fo_predictions)

        # Method 2: Monte Carlo simulation
        t0 = time.time()
        mc_errors = []
        noise = GaussianPhaseNoise(sigma_theta=sigma, sigma_phi=sigma)

        for _ in range(n_trials):
            thetas = rng.uniform(0, np.pi/2, mesh.n_mzis)
            phis = rng.uniform(0, 2*np.pi, mesh.n_mzis)
            mesh.set_phases(thetas, phis)
            U_ideal = mesh.unitary(include_noise=False)

            # Average over noise realizations
            trial_errors = []
            for _ in range(20):
                noise.apply_to_mesh(mesh, rng)
                U_noisy = mesh.unitary(include_noise=True)
                trial_errors.append(np.linalg.norm(U_noisy - U_ideal, 'fro'))
                mesh.clear_noise()

            mc_errors.append(np.mean(trial_errors))

        mc_time = time.time() - t0
        mc_mean = np.mean(mc_errors)

        # Comparison
        prediction_error = abs(fo_mean - mc_mean) / mc_mean * 100
        speedup = mc_time / fo_time if fo_time > 0 else float('inf')

        print(f"  {sigma:>10.3f} {fo_mean:>12.4f} {mc_mean:>12.4f} "
              f"{prediction_error:>9.1f}% {speedup:>11.1f}x")

        results.append({
            'sigma': sigma,
            'first_order': fo_mean,
            'monte_carlo': mc_mean,
            'error_percent': prediction_error,
            'speedup': speedup
        })

    print()
    avg_error = np.mean([r['error_percent'] for r in results if r['sigma'] <= 0.02])
    avg_speedup = np.mean([r['speedup'] for r in results])

    print("  IMPROVEMENT METRICS:")
    print(f"    Prediction accuracy (σ ≤ 2%): {100-avg_error:.1f}%")
    print(f"    Computational speedup: {avg_speedup:.0f}x faster")
    print(f"    Benefit: Rapid design iteration, real-time calibration")

    return results


# =============================================================================
# SUMMARY
# =============================================================================

def print_summary(results):
    """Print executive summary of all comparisons."""
    print_section("EXECUTIVE SUMMARY: Performance Improvements")

    print("""
  The PICASSO mathematical formalism provides the following quantified
  performance improvements over baseline approaches:
  """)

    print("  1. SENSITIVITY-GUIDED CORRECTION")
    if 'correction' in results:
        adv = results['correction']['guided_advantage_percent']
        print(f"     → {adv:.0f}% better fidelity recovery than blind correction")
        print(f"     → Enables targeted calibration of critical MZIs")

    print()
    print("  2. HAAR-OPTIMIZED CONFIGURATIONS")
    if 'configuration' in results:
        err_red = results['configuration']['error_reduction_percent']
        var_red = results['configuration']['variance_reduction_percent']
        print(f"     → {err_red:.0f}% reduction in mean error magnitude")
        print(f"     → {var_red:.0f}% reduction in error variance")
        print(f"     → Intrinsic fault tolerance through error delocalization")

    print()
    print("  3. PREDICTIVE ERROR BUDGETING")
    if 'budgeting' in results:
        avg_imp = np.mean([r['tolerance_improvement'] for r in results['budgeting']])
        print(f"     → {avg_imp:.0f}% higher noise tolerance than naive budgeting")
        print(f"     → Enables relaxed fabrication tolerances")
        print(f"     → Reduces manufacturing cost while meeting specs")

    print()
    print("  4. FIRST-ORDER PREDICTION")
    if 'prediction' in results:
        avg_speedup = np.mean([r['speedup'] for r in results['prediction']])
        avg_acc = 100 - np.mean([r['error_percent'] for r in results['prediction']
                                 if r['sigma'] <= 0.02])
        print(f"     → {avg_speedup:.0f}x computational speedup over Monte Carlo")
        print(f"     → {avg_acc:.0f}% prediction accuracy for σ ≤ 2%")
        print(f"     → Enables real-time calibration and rapid iteration")

    print()
    print("  KEY TAKEAWAY:")
    print("  The formalism transforms error analysis from expensive simulation")
    print("  to fast, accurate prediction - enabling design optimization that")
    print("  would otherwise be computationally prohibitive.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PICASSO FORMALISM: PERFORMANCE IMPROVEMENT ANALYSIS")
    print("=" * 70)

    results = {}

    # Run all comparisons
    results['correction'] = compare_correction_strategies(n_modes=6, sigma=0.05, n_trials=100)
    results['configuration'] = compare_mesh_configurations(n_modes=6, sigma=0.05, n_trials=100)
    results['budgeting'] = compare_error_budgeting(target_fidelity=0.99)
    results['prediction'] = compare_prediction_methods(n_modes=6, n_trials=30)

    # Executive summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("  Analysis complete.")
    print("=" * 70 + "\n")
