# PICASSO Simulation Framework

A Python framework for simulating and analyzing large-scale photonic Mach-Zehnder Interferometer (MZI) meshes, developed in support of the DARPA PICASSO program.

## Overview

This framework implements the mathematical concepts from the PICASSO ideas document:

- **Error Accumulation Analysis**: First-order perturbation theory for understanding how phase noise propagates through MZI meshes
- **Sensitivity Analysis**: Jacobian computation to identify sensitive vs. robust MZIs
- **Random Matrix Theory**: Haar random unitaries and Marchenko-Pastur analysis for condition number statistics
- **Fidelity Metrics**: Comprehensive metrics for comparing perturbed unitaries to targets

## Installation

```bash
# Ensure you have Python 3.8+ with NumPy and SciPy
pip install numpy scipy

# Clone or copy the picasso_sim directory
cd /path/to/DARPA_PICASSO
```

## Quick Start

```python
import numpy as np
from picasso_sim.core.mesh import ClementsMesh, random_mesh
from picasso_sim.core.noise import GaussianPhaseNoise
from picasso_sim.analysis.perturbation import PerturbationAnalyzer
from picasso_sim.analysis.fidelity import fidelity

# Create a 6-mode Clements mesh with random phases
rng = np.random.default_rng(42)
mesh = random_mesh(6, "clements", rng)

# Apply Gaussian phase noise
noise = GaussianPhaseNoise(sigma_theta=0.01, sigma_phi=0.01)
noise.apply_to_mesh(mesh, rng)

# Analyze perturbation effects
analyzer = PerturbationAnalyzer(mesh)
result = analyzer.analyze()

print(f"Fidelity loss: {result.fidelity_loss:.4f}")
print(f"Perturbation norm: {result.perturbation_norm:.4f}")
```

## Module Structure

```
picasso_sim/
├── core/
│   ├── mzi.py          # MZI unitary operators U(θ,φ)
│   ├── mesh.py         # Mesh architectures (Clements, Reck, Butterfly)
│   └── noise.py        # Noise models (Gaussian, thermal, fabrication)
├── analysis/
│   ├── perturbation.py # First-order perturbation theory
│   ├── sensitivity.py  # Jacobian computation, adjoint gradients
│   ├── fidelity.py     # Fidelity metrics
│   └── condition.py    # Condition number analysis
├── random_matrix/
│   ├── haar.py         # Haar random unitary generation
│   └── marchenko_pastur.py  # MP distribution analysis
├── inference/          # (Future) Bayesian noise localization
├── surrogates/         # (Future) Neural network surrogates
├── optimization/       # (Future) Adjoint-based calibration
└── examples/
    └── demo_error_analysis.py  # Demonstration script
```

## Running the Demo

```bash
cd /path/to/DARPA_PICASSO
python picasso_sim/examples/demo_error_analysis.py
```

## Key Concepts

### MZI Unitary
Each MZI implements a 2×2 unitary:
```
U(θ, φ) = [[cos(θ), i·sin(θ)], [i·sin(θ), cos(θ)]] · [[e^(iφ), 0], [0, 1]]
```

### Error Accumulation
For a mesh of L layers, the perturbed unitary is:
```
U_mesh = ∏(U_i + ΔU_i) ≈ U_target + ε·∑J_i·H_i
```
where J_i are Jacobian-like sensitivity operators.

### Scaling Law
Error scales with circuit depth:
```
||δU|| ∝ L^β  (β ≈ 0.5-1.0)
```

## Dependencies

- NumPy >= 1.20
- SciPy >= 1.7

## License

Developed for DARPA PICASSO program. See program documentation for usage terms.
