"""
Noise models for photonic MZI meshes.

This module provides various noise models that capture real-world
imperfections in photonic circuits:

1. Gaussian Phase Noise: Random phase errors from control electronics
2. Thermal Drift: Slow temporal variations from temperature changes
3. Fabrication Noise: Static errors from manufacturing variations
4. Crosstalk: Inter-MZI coupling effects

These models can be composed to simulate realistic operating conditions.
"""

import numpy as np
from typing import Optional, Tuple, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass


class NoiseModel(ABC):
    """
    Abstract base class for noise models.

    Noise models generate perturbations (δθ, δφ) for MZI phases
    that can be applied to simulate realistic imperfections.
    """

    @abstractmethod
    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample noise perturbations for n MZIs.

        Parameters
        ----------
        n_mzis : int
            Number of MZIs to generate noise for.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        delta_thetas : np.ndarray
            Phase perturbations for θ (internal phase).
        delta_phis : np.ndarray
            Phase perturbations for φ (external phase).
        """
        pass

    def apply_to_mesh(self, mesh, rng: Optional[np.random.Generator] = None):
        """
        Apply noise to a photonic mesh.

        Parameters
        ----------
        mesh : PhotonicMesh
            The mesh to apply noise to.
        rng : np.random.Generator, optional
            Random number generator.
        """
        delta_thetas, delta_phis = self.sample(mesh.n_mzis, rng)
        mesh.apply_noise(delta_thetas, delta_phis)


@dataclass
class GaussianPhaseNoise(NoiseModel):
    """
    Independent Gaussian noise on phase shifters.

    Models random phase errors from:
    - DAC quantization
    - Electronic noise in drivers
    - Shot noise in feedback systems

    Parameters
    ----------
    sigma_theta : float
        Standard deviation of θ noise (radians).
    sigma_phi : float
        Standard deviation of φ noise (radians).
    correlated : bool
        If True, θ and φ noise are correlated.
    correlation : float
        Correlation coefficient if correlated=True.
    """
    sigma_theta: float = 0.01  # ~0.6 degrees
    sigma_phi: float = 0.01
    correlated: bool = False
    correlation: float = 0.0

    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        if self.correlated and self.correlation != 0:
            # Generate correlated noise
            cov = np.array([
                [self.sigma_theta**2, self.correlation * self.sigma_theta * self.sigma_phi],
                [self.correlation * self.sigma_theta * self.sigma_phi, self.sigma_phi**2]
            ])
            mean = [0, 0]
            samples = rng.multivariate_normal(mean, cov, size=n_mzis)
            delta_thetas = samples[:, 0]
            delta_phis = samples[:, 1]
        else:
            delta_thetas = rng.normal(0, self.sigma_theta, n_mzis)
            delta_phis = rng.normal(0, self.sigma_phi, n_mzis)

        return delta_thetas, delta_phis

    def __repr__(self) -> str:
        return f"GaussianPhaseNoise(σ_θ={self.sigma_theta:.4f}, σ_φ={self.sigma_phi:.4f})"


@dataclass
class ThermalDriftNoise(NoiseModel):
    """
    Thermal drift noise model.

    Models slow temporal variations in phase due to temperature changes.
    Uses a random walk or Ornstein-Uhlenbeck process.

    Parameters
    ----------
    drift_rate : float
        Rate of thermal drift (radians per time unit).
    temperature_sensitivity : float
        Phase change per degree Celsius (rad/°C).
    spatial_correlation_length : float
        Length scale over which thermal noise is correlated (in MZI indices).
    """
    drift_rate: float = 0.001
    temperature_sensitivity: float = 0.01  # rad/°C, typical for silicon
    spatial_correlation_length: float = 5.0

    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Generate spatially correlated noise
        # Use exponential correlation: C(i,j) = exp(-|i-j|/L)
        indices = np.arange(n_mzis)
        dist_matrix = np.abs(indices[:, None] - indices[None, :])
        correlation_matrix = np.exp(-dist_matrix / self.spatial_correlation_length)

        # Cholesky decomposition for correlated sampling
        L = np.linalg.cholesky(correlation_matrix + 1e-10 * np.eye(n_mzis))

        # Sample correlated Gaussian
        z = rng.standard_normal(n_mzis)
        correlated_noise = L @ z

        # Scale by drift rate
        delta_thetas = self.drift_rate * correlated_noise
        delta_phis = self.drift_rate * correlated_noise * 0.5  # φ less sensitive

        return delta_thetas, delta_phis

    def sample_temporal(self, n_mzis: int, n_timesteps: int,
                        rng: Optional[np.random.Generator] = None
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample time-varying thermal drift (Ornstein-Uhlenbeck process).

        Returns arrays of shape (n_timesteps, n_mzis).
        """
        if rng is None:
            rng = np.random.default_rng()

        theta = 0.1  # Mean reversion rate
        mu = 0  # Long-term mean
        sigma = self.drift_rate

        delta_thetas = np.zeros((n_timesteps, n_mzis))
        delta_phis = np.zeros((n_timesteps, n_mzis))

        # Initial state
        delta_thetas[0] = rng.normal(0, sigma, n_mzis)
        delta_phis[0] = rng.normal(0, sigma * 0.5, n_mzis)

        # Evolve OU process
        dt = 1.0
        for t in range(1, n_timesteps):
            dW_theta = rng.normal(0, np.sqrt(dt), n_mzis)
            dW_phi = rng.normal(0, np.sqrt(dt), n_mzis)

            delta_thetas[t] = (delta_thetas[t-1] +
                              theta * (mu - delta_thetas[t-1]) * dt +
                              sigma * dW_theta)
            delta_phis[t] = (delta_phis[t-1] +
                            theta * (mu - delta_phis[t-1]) * dt +
                            sigma * 0.5 * dW_phi)

        return delta_thetas, delta_phis


@dataclass
class FabricationNoise(NoiseModel):
    """
    Static fabrication noise model.

    Models permanent errors from manufacturing:
    - Waveguide width variations
    - Coupler imbalance
    - Path length differences

    These errors are static (don't change over time) but vary across MZIs.

    Parameters
    ----------
    sigma_theta : float
        Standard deviation of θ errors from coupler imbalance.
    sigma_phi : float
        Standard deviation of φ errors from path length variations.
    bias_theta : float
        Systematic bias in θ (all MZIs shifted).
    bias_phi : float
        Systematic bias in φ.
    """
    sigma_theta: float = 0.02  # ~1.1 degrees
    sigma_phi: float = 0.05  # ~2.9 degrees
    bias_theta: float = 0.0
    bias_phi: float = 0.0

    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        delta_thetas = self.bias_theta + rng.normal(0, self.sigma_theta, n_mzis)
        delta_phis = self.bias_phi + rng.normal(0, self.sigma_phi, n_mzis)

        return delta_thetas, delta_phis


@dataclass
class CrosstalkNoise(NoiseModel):
    """
    Crosstalk noise model.

    Models inter-MZI coupling effects where the phase of one MZI
    affects neighboring MZIs (e.g., through thermal or electrical crosstalk).

    Parameters
    ----------
    coupling_strength : float
        Strength of nearest-neighbor coupling.
    coupling_range : int
        Number of neighboring MZIs affected.
    """
    coupling_strength: float = 0.005
    coupling_range: int = 2

    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        # Base noise
        base_theta = rng.normal(0, 0.01, n_mzis)
        base_phi = rng.normal(0, 0.01, n_mzis)

        # Add crosstalk contributions
        delta_thetas = base_theta.copy()
        delta_phis = base_phi.copy()

        for i in range(n_mzis):
            for j in range(max(0, i - self.coupling_range),
                          min(n_mzis, i + self.coupling_range + 1)):
                if i != j:
                    distance = abs(i - j)
                    coupling = self.coupling_strength / distance
                    delta_thetas[i] += coupling * base_theta[j]
                    delta_phis[i] += coupling * base_phi[j]

        return delta_thetas, delta_phis


class CompositeNoise(NoiseModel):
    """
    Composite noise model combining multiple noise sources.

    Parameters
    ----------
    models : list of NoiseModel
        Noise models to combine.
    weights : list of float, optional
        Weights for each model. Default is equal weights.
    """

    def __init__(self, models: List[NoiseModel], weights: Optional[List[float]] = None):
        self.models = models
        if weights is None:
            weights = [1.0] * len(models)
        self.weights = np.array(weights) / sum(weights)

    def sample(self, n_mzis: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        if rng is None:
            rng = np.random.default_rng()

        delta_thetas = np.zeros(n_mzis)
        delta_phis = np.zeros(n_mzis)

        for model, weight in zip(self.models, self.weights):
            dt, dp = model.sample(n_mzis, rng)
            delta_thetas += weight * dt
            delta_phis += weight * dp

        return delta_thetas, delta_phis


def realistic_noise_model(scenario: str = "typical") -> NoiseModel:
    """
    Create a realistic noise model for common scenarios.

    Parameters
    ----------
    scenario : str
        One of:
        - "ideal": No noise
        - "low": Low noise (well-controlled lab environment)
        - "typical": Typical operating conditions
        - "high": High noise (challenging conditions)
        - "fabrication_dominated": Mainly static fabrication errors

    Returns
    -------
    NoiseModel
        Configured noise model.
    """
    if scenario == "ideal":
        return GaussianPhaseNoise(sigma_theta=0, sigma_phi=0)

    elif scenario == "low":
        return CompositeNoise([
            GaussianPhaseNoise(sigma_theta=0.005, sigma_phi=0.005),
            FabricationNoise(sigma_theta=0.01, sigma_phi=0.02),
        ])

    elif scenario == "typical":
        return CompositeNoise([
            GaussianPhaseNoise(sigma_theta=0.01, sigma_phi=0.01),
            ThermalDriftNoise(drift_rate=0.005),
            FabricationNoise(sigma_theta=0.02, sigma_phi=0.05),
        ])

    elif scenario == "high":
        return CompositeNoise([
            GaussianPhaseNoise(sigma_theta=0.03, sigma_phi=0.03),
            ThermalDriftNoise(drift_rate=0.02),
            FabricationNoise(sigma_theta=0.05, sigma_phi=0.1),
            CrosstalkNoise(coupling_strength=0.01),
        ])

    elif scenario == "fabrication_dominated":
        return FabricationNoise(sigma_theta=0.05, sigma_phi=0.1)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")


@dataclass
class NoiseStatistics:
    """
    Container for noise statistics computed from samples.
    """
    mean_theta: float
    mean_phi: float
    std_theta: float
    std_phi: float
    max_theta: float
    max_phi: float
    correlation: float

    @classmethod
    def from_samples(cls, delta_thetas: np.ndarray, delta_phis: np.ndarray) -> "NoiseStatistics":
        """Compute statistics from noise samples."""
        return cls(
            mean_theta=np.mean(delta_thetas),
            mean_phi=np.mean(delta_phis),
            std_theta=np.std(delta_thetas),
            std_phi=np.std(delta_phis),
            max_theta=np.max(np.abs(delta_thetas)),
            max_phi=np.max(np.abs(delta_phis)),
            correlation=np.corrcoef(delta_thetas, delta_phis)[0, 1] if len(delta_thetas) > 1 else 0
        )
