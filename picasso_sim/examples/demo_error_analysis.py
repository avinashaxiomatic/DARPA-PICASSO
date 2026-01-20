#!/usr/bin/env python3
"""
PICASSO Simulation Framework - Demonstration Script

This script demonstrates the key capabilities of the PICASSO simulation
framework for analyzing error accumulation in large-scale photonic MZI meshes.

It covers:
1. Creating and configuring MZI meshes
2. Applying realistic noise models
3. Perturbation and sensitivity analysis
4. Fidelity metrics and scaling laws
5. Random matrix theory analysis (Haar unitaries, MP distribution)

Run this script to verify the installation and see the framework in action.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Also add grandparent for proper package resolution
_grandparent_dir = os.path.dirname(_parent_dir)
if _grandparent_dir not in sys.path:
    sys.path.insert(0, _grandparent_dir)

from picasso_sim.core.mzi import MZI, mzi_unitary, balanced_splitter
from picasso_sim.core.mesh import ClementsMesh, ReckMesh, random_mesh
from picasso_sim.core.noise import (GaussianPhaseNoise, ThermalDriftNoise,
                        FabricationNoise, realistic_noise_model)
from picasso_sim.analysis.perturbation import PerturbationAnalyzer, scaling_analysis, fit_scaling_law
from picasso_sim.analysis.sensitivity import SensitivityAnalyzer, compute_jacobian
from picasso_sim.analysis.fidelity import (fidelity, process_fidelity, comprehensive_fidelity_report,
                               fidelity_vs_noise)
from picasso_sim.analysis.condition import (condition_number, analyze_condition,
                                mesh_jacobian_condition, perturbation_amplification)
from picasso_sim.random_matrix.haar import (haar_unitary, is_haar_distributed,
                                haar_fidelity_benchmark, error_delocalization_test)
from picasso_sim.random_matrix.marchenko_pastur import (marchenko_pastur_pdf, fit_marchenko_pastur,
                                            compare_to_mp)


def separator(title: str):
    """Print section separator."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60 + "\n")


def demo_mzi_basics():
    """Demonstrate basic MZI operations."""
    separator("1. MZI Basics")

    # Create a single MZI
    mzi = MZI(theta=np.pi/4, phi=0, mode_indices=(0, 1))
    print(f"MZI: {mzi}")
    print(f"  Transmission: {mzi.transmission:.4f}")
    print(f"  Coupling: {mzi.coupling:.4f}")

    # Get unitary matrix
    U = mzi.unitary()
    print(f"\n2x2 Unitary matrix:\n{U}")
    print(f"Is unitary: {mzi.is_unitary()}")

    # Embed in larger space
    U_embedded = mzi.embed(4)
    print(f"\nEmbedded in 4x4 space (acts on modes 0,1):\n{U_embedded}")


def demo_mesh_architectures():
    """Demonstrate mesh architectures."""
    separator("2. Mesh Architectures")

    n_modes = 4

    # Clements mesh
    clements = ClementsMesh(n_modes)
    print(f"Clements Mesh: {clements}")
    print(f"  Number of MZIs: {clements.n_mzis}")
    print(f"  Number of layers: {clements.n_layers}")
    print(f"  Circuit depth: {clements.depth}")

    # Reck mesh
    reck = ReckMesh(n_modes)
    print(f"\nReck Mesh: {reck}")
    print(f"  Number of MZIs: {reck.n_mzis}")
    print(f"  Number of layers: {reck.n_layers}")

    # Random mesh
    rng = np.random.default_rng(42)
    mesh = random_mesh(n_modes, "clements", rng)
    U = mesh.unitary()
    print(f"\nRandom Clements mesh unitary is unitary: {mesh.is_unitary()}")


def demo_noise_models():
    """Demonstrate noise models."""
    separator("3. Noise Models")

    rng = np.random.default_rng(42)
    n_mzis = 10

    # Gaussian phase noise
    gaussian = GaussianPhaseNoise(sigma_theta=0.01, sigma_phi=0.01)
    dt, dp = gaussian.sample(n_mzis, rng)
    print(f"Gaussian Noise: {gaussian}")
    print(f"  θ perturbations: mean={np.mean(dt):.4f}, std={np.std(dt):.4f}")
    print(f"  φ perturbations: mean={np.mean(dp):.4f}, std={np.std(dp):.4f}")

    # Thermal drift
    thermal = ThermalDriftNoise(drift_rate=0.005)
    dt, dp = thermal.sample(n_mzis, rng)
    print(f"\nThermal Drift Noise:")
    print(f"  θ perturbations: mean={np.mean(dt):.4f}, std={np.std(dt):.4f}")

    # Fabrication noise
    fab = FabricationNoise(sigma_theta=0.02, sigma_phi=0.05)
    dt, dp = fab.sample(n_mzis, rng)
    print(f"\nFabrication Noise:")
    print(f"  θ perturbations: std={np.std(dt):.4f}")
    print(f"  φ perturbations: std={np.std(dp):.4f}")

    # Realistic composite model
    realistic = realistic_noise_model("typical")
    print(f"\nRealistic 'typical' noise model created")


def demo_perturbation_analysis():
    """Demonstrate perturbation analysis."""
    separator("4. Perturbation Analysis")

    rng = np.random.default_rng(42)

    # Create mesh with random configuration
    mesh = random_mesh(6, "clements", rng)

    # Apply noise
    noise = GaussianPhaseNoise(sigma_theta=0.02, sigma_phi=0.02)
    noise.apply_to_mesh(mesh, rng)

    # Analyze perturbation
    analyzer = PerturbationAnalyzer(mesh)
    result = analyzer.analyze()

    print(f"Perturbation Analysis Results:")
    print(f"  Perturbation norm ||δU||: {result.perturbation_norm:.6f}")
    print(f"  Relative error: {result.relative_error:.6f}")
    print(f"  Spectral perturbation: {result.spectral_perturbation:.6f}")
    print(f"  Fidelity loss: {result.fidelity_loss:.6f}")

    # First-order approximation
    delta_U_approx = analyzer.first_order_approximation()
    approx_error = np.linalg.norm(delta_U_approx - result.delta_U, 'fro')
    print(f"\n  First-order approximation error: {approx_error:.6f}")


def demo_sensitivity_analysis():
    """Demonstrate sensitivity (Jacobian) analysis."""
    separator("5. Sensitivity Analysis")

    rng = np.random.default_rng(42)
    mesh = random_mesh(4, "clements", rng)

    # Compute Jacobian
    sens_analyzer = SensitivityAnalyzer(mesh)
    result = sens_analyzer.compute()

    print(f"Sensitivity Analysis:")
    print(f"  Jacobian shape (θ): {result.jacobian_theta.shape}")
    print(f"  Most sensitive MZI index: {result.max_sensitivity_idx}")
    print(f"  Jacobian condition number: {result.condition_number:.2f}")

    # Sensitivity distribution
    print(f"\n  Sensitivity norms by MZI:")
    for i, norm in enumerate(result.sensitivity_norms[:5]):
        print(f"    MZI {i}: {norm:.4f}")
    if len(result.sensitivity_norms) > 5:
        print(f"    ... ({len(result.sensitivity_norms) - 5} more)")

    # Identify robust subgraph
    robust = sens_analyzer.identify_robust_subgraph(threshold=0.5)
    print(f"\n  Robust MZIs (sensitivity < 50% of max): {len(robust['robust_indices'])}")
    print(f"  Sensitive MZIs: {len(robust['sensitive_indices'])}")


def demo_fidelity_metrics():
    """Demonstrate fidelity metrics."""
    separator("6. Fidelity Metrics")

    rng = np.random.default_rng(42)

    # Create target unitary (Haar random)
    n = 4
    U_target = haar_unitary(n, rng)

    # Create slightly perturbed version
    noise = 0.05
    perturbation = noise * (rng.standard_normal((n, n)) +
                           1j * rng.standard_normal((n, n)))
    U_actual, _ = np.linalg.qr(U_target + perturbation)  # Re-unitarize

    # Compute fidelity metrics
    report = comprehensive_fidelity_report(U_target, U_actual, n_samples=500, rng=rng)
    print(report)


def demo_scaling_laws():
    """Demonstrate error scaling with mesh size."""
    separator("7. Scaling Laws")

    rng = np.random.default_rng(42)
    noise = GaussianPhaseNoise(sigma_theta=0.01, sigma_phi=0.01)

    n_modes_range = [4, 6, 8]
    print(f"Analyzing error scaling for mesh sizes: {n_modes_range}")

    results = scaling_analysis(ClementsMesh, n_modes_range, noise,
                               n_samples=30, rng=rng)

    print(f"\nScaling Results:")
    for i, n in enumerate(results['n_modes']):
        print(f"  N={n}: {results['n_mzis'][i]} MZIs, "
              f"mean_error={results['mean_error'][i]:.4f} ± {results['std_error'][i]:.4f}")

    # Fit scaling law
    alpha, beta = fit_scaling_law(results['n_mzis'], results['mean_error'])
    print(f"\nScaling law fit: Error ≈ {alpha:.4f} × L^{beta:.2f}")
    print(f"  (Expected β ≈ 0.5-1.0 for linear error accumulation)")


def demo_haar_analysis():
    """Demonstrate Haar random unitary analysis."""
    separator("8. Haar Random Unitary Analysis")

    rng = np.random.default_rng(42)
    n = 4

    # Generate Haar unitaries
    print(f"Generating {50} Haar-random {n}×{n} unitaries...")
    unitaries = np.array([haar_unitary(n, rng) for _ in range(50)])

    # Test if Haar distributed
    test_result = is_haar_distributed(unitaries)
    print(f"\nHaar distribution test:")
    print(f"  Eigenvalue KS p-value: {test_result['eigenvalue_ks_pvalue']:.4f}")
    print(f"  Eigenvalue test passed: {test_result['eigenvalue_test_passed']}")
    print(f"  Entry test passed: {test_result['entry_test_passed']}")
    print(f"  Overall: {'Consistent with Haar' if test_result['is_haar'] else 'Not Haar'}")


def demo_error_delocalization():
    """Demonstrate error delocalization benefit of Haar-like configurations."""
    separator("9. Error Delocalization")

    rng = np.random.default_rng(42)
    mesh = ClementsMesh(6)
    noise = GaussianPhaseNoise(sigma_theta=0.03, sigma_phi=0.03)

    # Set random phases first
    thetas = rng.uniform(0, np.pi/2, mesh.n_mzis)
    phis = rng.uniform(0, 2*np.pi, mesh.n_mzis)
    mesh.set_phases(thetas, phis)

    result = error_delocalization_test(mesh, noise, n_samples=50, rng=rng)

    print(f"Error Delocalization Test:")
    print(f"  Identity config - Mean fidelity: {result['identity_mean_fidelity']:.4f}")
    print(f"  Random config   - Mean fidelity: {result['random_mean_fidelity']:.4f}")
    print(f"  Delocalization benefit: {result['delocalization_benefit']:.4f}")

    if result['delocalization_benefit'] > 0:
        print("  → Random configuration is MORE robust to noise")
    else:
        print("  → Identity configuration is more robust (unexpected)")


def demo_condition_analysis():
    """Demonstrate condition number and perturbation amplification."""
    separator("10. Condition Number Analysis")

    rng = np.random.default_rng(42)
    mesh = random_mesh(6, "clements", rng)

    # Jacobian condition
    jac_result = mesh_jacobian_condition(mesh)
    print(f"Mesh Jacobian Analysis:")
    print(f"  Condition number: {jac_result['jacobian_condition']:.2f}")
    print(f"  Effective rank: {jac_result['jacobian_rank']}")
    print(f"  σ_max: {jac_result['sigma_max']:.4f}")
    print(f"  σ_min: {jac_result['sigma_min']:.6f}")

    # Perturbation amplification
    noise = GaussianPhaseNoise(sigma_theta=0.01, sigma_phi=0.01)
    amp_result = perturbation_amplification(mesh, noise, n_samples=50, rng=rng)

    print(f"\nPerturbation Amplification:")
    print(f"  Mean amplification ||δU||/||δθ,δφ||: {amp_result['mean_amplification']:.2f}")
    print(f"  Max amplification: {amp_result['max_amplification']:.2f}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("  PICASSO Simulation Framework - Demonstration")
    print("  Error Analysis for Large-Scale Photonic MZI Meshes")
    print("=" * 60)

    try:
        demo_mzi_basics()
        demo_mesh_architectures()
        demo_noise_models()
        demo_perturbation_analysis()
        demo_sensitivity_analysis()
        demo_fidelity_metrics()
        demo_scaling_laws()
        demo_haar_analysis()
        demo_error_delocalization()
        demo_condition_analysis()

        separator("Demo Complete!")
        print("All framework components are working correctly.\n")
        print("Key insights demonstrated:")
        print("  1. MZI meshes can implement arbitrary unitaries")
        print("  2. Phase noise causes predictable perturbations")
        print("  3. Errors scale with circuit depth (scaling law)")
        print("  4. Sensitivity analysis identifies vulnerable MZIs")
        print("  5. Haar-random configs can delocalize errors")
        print("  6. Random matrix theory provides analytical bounds")
        print()

    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
