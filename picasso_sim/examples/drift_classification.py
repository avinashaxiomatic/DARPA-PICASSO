"""
Comprehensive Classification of Drift Mechanisms in Photonic Systems

Understanding drift types is crucial for:
1. Designing appropriate calibration strategies
2. Choosing measurement frequencies
3. Selecting materials and packaging
4. Predicting system lifetime
"""

import numpy as np
import sys

sys.path.insert(0, '.')

from picasso_sim.core.mesh import ClementsMesh
from picasso_sim.analysis.fidelity import fidelity


# =============================================================================
# DRIFT CLASSIFICATION
# =============================================================================

DRIFT_TAXONOMY = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    DRIFT MECHANISMS IN PHOTONIC SYSTEMS                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  1. THERMAL DRIFT                                                             ║
║     ├── 1.1 Ambient temperature change                                        ║
║     ├── 1.2 Self-heating (optical absorption)                                 ║
║     ├── 1.3 Joule heating (from phase shifters)                              ║
║     └── 1.4 Thermal crosstalk between MZIs                                    ║
║                                                                               ║
║  2. MECHANICAL DRIFT                                                          ║
║     ├── 2.1 Vibration-induced path length changes                            ║
║     ├── 2.2 Stress relaxation (packaging)                                     ║
║     ├── 2.3 Fiber coupling drift                                              ║
║     └── 2.4 Thermal expansion mismatch                                        ║
║                                                                               ║
║  3. MATERIAL AGING                                                            ║
║     ├── 3.1 Waveguide core densification                                      ║
║     ├── 3.2 Dopant diffusion                                                  ║
║     ├── 3.3 Hydrogen migration (silica)                                       ║
║     └── 3.4 Electromigration (metal heaters)                                  ║
║                                                                               ║
║  4. ENVIRONMENTAL                                                             ║
║     ├── 4.1 Humidity absorption                                               ║
║     ├── 4.2 Pressure changes (air gaps)                                       ║
║     ├── 4.3 Contamination/particles                                           ║
║     └── 4.4 Radiation damage (space applications)                             ║
║                                                                               ║
║  5. SOURCE-RELATED                                                            ║
║     ├── 5.1 Laser wavelength drift                                            ║
║     ├── 5.2 Power fluctuations                                                ║
║     ├── 5.3 Mode hopping                                                      ║
║     └── 5.4 Polarization drift                                                ║
║                                                                               ║
║  6. ELECTRONIC                                                                ║
║     ├── 6.1 DAC drift                                                         ║
│     ├── 6.2 Driver amplifier drift                                            ║
║     ├── 6.3 Reference voltage drift                                           ║
║     └── 6.4 Ground loops / EMI                                                ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


class DriftModel:
    """Base class for drift models."""

    def __init__(self, n_mzis, rng=None):
        self.n_mzis = n_mzis
        self.rng = rng or np.random.default_rng()
        self.time = 0
        self.cumulative_drift = np.zeros(n_mzis)

    def step(self, dt=1.0):
        """Advance time by dt and return new drift."""
        raise NotImplementedError

    def reset(self):
        """Reset drift to zero."""
        self.time = 0
        self.cumulative_drift = np.zeros(self.n_mzis)


# =============================================================================
# 1. THERMAL DRIFT MODELS
# =============================================================================

class AmbientThermalDrift(DriftModel):
    """
    1.1 Ambient Temperature Change

    Characteristics:
    - Timescale: Minutes to hours
    - Spatial pattern: Uniform or slow gradient across chip
    - Magnitude: ~0.1 rad/°C for silicon (dn/dT = 1.8e-4 /K)
    - Compressibility: HIGH (very smooth)

    Mitigation:
    - Temperature stabilization (TEC)
    - Athermal waveguide design
    - Periodic recalibration
    """

    def __init__(self, n_mzis, temp_sensitivity=0.1, temp_noise=0.01, rng=None):
        super().__init__(n_mzis, rng)
        self.temp_sensitivity = temp_sensitivity  # rad/°C
        self.temp_noise = temp_noise  # °C/step
        self.current_temp = 0  # Deviation from setpoint

    def step(self, dt=1.0):
        # Slow random walk in temperature
        self.current_temp += self.rng.normal(0, self.temp_noise * np.sqrt(dt))

        # Uniform phase shift across all MZIs
        new_drift = np.ones(self.n_mzis) * self.temp_sensitivity * self.current_temp
        delta = new_drift - self.cumulative_drift
        self.cumulative_drift = new_drift
        self.time += dt
        return delta


class SelfHeatingDrift(DriftModel):
    """
    1.2 Self-Heating (Optical Absorption)

    Characteristics:
    - Timescale: Milliseconds to seconds
    - Spatial pattern: Localized at high-power regions
    - Magnitude: Depends on optical power and absorption
    - Compressibility: MEDIUM (localized hotspots)

    Mitigation:
    - Lower optical power
    - Better heat sinking
    - Power-aware phase compensation
    """

    def __init__(self, n_mzis, absorption_coeff=0.001, thermal_tau=0.1, rng=None):
        super().__init__(n_mzis, rng)
        self.absorption = absorption_coeff
        self.tau = thermal_tau  # Thermal time constant
        self.power_distribution = self.rng.exponential(1, n_mzis)  # Power varies by position
        self.power_distribution /= np.mean(self.power_distribution)

    def step(self, dt=1.0):
        # Heating proportional to local power
        heating = self.power_distribution * self.absorption

        # First-order thermal response
        alpha = 1 - np.exp(-dt / self.tau)
        target_drift = heating + self.rng.normal(0, 0.001, self.n_mzis)
        self.cumulative_drift = (1 - alpha) * self.cumulative_drift + alpha * target_drift
        self.time += dt
        return self.cumulative_drift.copy()


class ThermalCrosstalkDrift(DriftModel):
    """
    1.4 Thermal Crosstalk Between MZIs

    Characteristics:
    - Timescale: Seconds
    - Spatial pattern: Decays with distance from active MZI
    - Magnitude: ~1-10% of direct heating
    - Compressibility: HIGH (smooth spatial decay)

    Mitigation:
    - Thermal isolation trenches
    - Crosstalk compensation matrix
    - Sparse activation patterns
    """

    def __init__(self, n_mzis, crosstalk_length=10, crosstalk_strength=0.05, rng=None):
        super().__init__(n_mzis, rng)
        self.crosstalk_length = crosstalk_length
        self.crosstalk_strength = crosstalk_strength

        # Build crosstalk matrix
        positions = np.arange(n_mzis)
        dist = np.abs(positions[:, None] - positions[None, :])
        self.crosstalk_matrix = crosstalk_strength * np.exp(-dist / crosstalk_length)
        np.fill_diagonal(self.crosstalk_matrix, 1)

    def step(self, dt=1.0):
        # Random activation pattern
        activation = self.rng.random(self.n_mzis)

        # Crosstalk spreads heat
        heating = self.crosstalk_matrix @ activation
        self.cumulative_drift += heating * 0.01 * dt
        self.time += dt
        return self.cumulative_drift.copy()


# =============================================================================
# 2. MECHANICAL DRIFT MODELS
# =============================================================================

class VibrationDrift(DriftModel):
    """
    2.1 Vibration-Induced Path Length Changes

    Characteristics:
    - Timescale: Milliseconds (acoustic frequencies)
    - Spatial pattern: Mode shapes of chip/package
    - Magnitude: Sub-nm path changes = mrad phase
    - Compressibility: HIGH (few mechanical modes)

    Mitigation:
    - Vibration isolation
    - Stiff mounting
    - Fast feedback (kHz rates)
    """

    def __init__(self, n_mzis, n_modes=3, freq_range=(10, 1000), rng=None):
        super().__init__(n_mzis, rng)
        self.n_modes = n_modes

        # Random mechanical mode shapes
        self.mode_shapes = []
        self.frequencies = []
        self.phases = []

        for _ in range(n_modes):
            # Mode shape: sinusoidal across chip
            k = self.rng.integers(1, 10)
            shape = np.sin(2 * np.pi * k * np.arange(n_mzis) / n_mzis + self.rng.uniform(0, 2*np.pi))
            self.mode_shapes.append(shape)
            self.frequencies.append(self.rng.uniform(*freq_range))
            self.phases.append(self.rng.uniform(0, 2*np.pi))

        self.amplitudes = self.rng.exponential(0.001, n_modes)

    def step(self, dt=1.0):
        self.time += dt
        drift = np.zeros(self.n_mzis)

        for i in range(self.n_modes):
            drift += (self.amplitudes[i] * self.mode_shapes[i] *
                     np.sin(2 * np.pi * self.frequencies[i] * self.time + self.phases[i]))

        self.cumulative_drift = drift
        return drift


class StressRelaxationDrift(DriftModel):
    """
    2.2 Stress Relaxation (Packaging)

    Characteristics:
    - Timescale: Hours to days (logarithmic)
    - Spatial pattern: Concentrated near package interfaces
    - Magnitude: Can be large initially, decays over time
    - Compressibility: MEDIUM

    Mitigation:
    - Burn-in period
    - Stress-free packaging
    - Periodic recalibration (less frequent over time)
    """

    def __init__(self, n_mzis, initial_stress=0.1, relaxation_tau=1000, rng=None):
        super().__init__(n_mzis, rng)
        self.initial_stress = initial_stress
        self.tau = relaxation_tau

        # Stress concentrated at edges
        positions = np.arange(n_mzis) / n_mzis
        self.stress_profile = np.exp(-10 * positions) + np.exp(-10 * (1 - positions))
        self.stress_profile /= np.max(self.stress_profile)

    def step(self, dt=1.0):
        self.time += dt

        # Logarithmic relaxation
        drift = self.initial_stress * self.stress_profile * np.log(1 + self.time) / np.log(1 + self.tau)
        delta = drift - self.cumulative_drift
        self.cumulative_drift = drift
        return delta


# =============================================================================
# 3. MATERIAL AGING MODELS
# =============================================================================

class WaveguideDensificationDrift(DriftModel):
    """
    3.1 Waveguide Core Densification

    Characteristics:
    - Timescale: Months to years
    - Spatial pattern: Uniform (material property)
    - Magnitude: Small but irreversible
    - Compressibility: VERY HIGH (uniform)

    Mitigation:
    - Pre-annealing
    - Lifetime characterization
    - Drift prediction models
    """

    def __init__(self, n_mzis, densification_rate=1e-6, rng=None):
        super().__init__(n_mzis, rng)
        self.rate = densification_rate  # rad/hour

    def step(self, dt=1.0):
        self.time += dt
        # Uniform densification
        drift_increment = np.ones(self.n_mzis) * self.rate * dt
        self.cumulative_drift += drift_increment
        return drift_increment


class ElectromigrationDrift(DriftModel):
    """
    3.4 Electromigration (Metal Heaters)

    Characteristics:
    - Timescale: Months to years
    - Spatial pattern: Depends on heater usage
    - Magnitude: Gradual heater resistance change
    - Compressibility: LOW (random per heater)

    Mitigation:
    - Lower current density
    - Better metallization
    - Usage-aware lifetime management
    """

    def __init__(self, n_mzis, migration_rate=1e-7, rng=None):
        super().__init__(n_mzis, rng)
        self.rate = migration_rate

        # Usage varies per MZI
        self.usage = self.rng.exponential(1, n_mzis)

    def step(self, dt=1.0):
        self.time += dt
        # Drift proportional to usage
        drift_increment = self.usage * self.rate * dt * self.rng.uniform(0.5, 1.5, self.n_mzis)
        self.cumulative_drift += drift_increment
        return drift_increment


# =============================================================================
# 4. ENVIRONMENTAL DRIFT MODELS
# =============================================================================

class HumidityDrift(DriftModel):
    """
    4.1 Humidity Absorption

    Characteristics:
    - Timescale: Hours (diffusion-limited)
    - Spatial pattern: Edge-to-center gradient
    - Magnitude: Depends on cladding material
    - Compressibility: HIGH (smooth gradient)

    Mitigation:
    - Hermetic packaging
    - Hydrophobic coatings
    - Humidity control
    """

    def __init__(self, n_mzis, humidity_sensitivity=0.01, diffusion_rate=0.01, rng=None):
        super().__init__(n_mzis, rng)
        self.sensitivity = humidity_sensitivity
        self.diffusion = diffusion_rate

        # Humidity enters from edges
        positions = np.arange(n_mzis) / n_mzis
        self.penetration_profile = np.exp(-5 * positions) + np.exp(-5 * (1 - positions))

    def step(self, dt=1.0):
        self.time += dt

        # Humidity random walk
        humidity_change = self.rng.normal(0, 0.01 * np.sqrt(dt))

        # Diffusion into chip
        drift = self.sensitivity * humidity_change * self.penetration_profile
        self.cumulative_drift += drift
        return drift


# =============================================================================
# 5. SOURCE-RELATED DRIFT MODELS
# =============================================================================

class WavelengthDrift(DriftModel):
    """
    5.1 Laser Wavelength Drift

    Characteristics:
    - Timescale: Seconds to minutes
    - Spatial pattern: Wavelength-dependent (varies with MZI design)
    - Magnitude: ~1 rad per nm wavelength shift
    - Compressibility: HIGH (uniform or systematic)

    Mitigation:
    - Wavelength locking
    - Wavelength-insensitive designs
    - Wavelength monitoring and compensation
    """

    def __init__(self, n_mzis, wavelength_sensitivity=1.0, wavelength_noise=0.001, rng=None):
        super().__init__(n_mzis, rng)
        self.sensitivity = wavelength_sensitivity  # rad/nm
        self.noise = wavelength_noise  # nm/step

        # Different MZIs have different wavelength sensitivity
        # (depends on path length difference)
        self.mzi_sensitivity = 1 + 0.1 * self.rng.normal(0, 1, n_mzis)

    def step(self, dt=1.0):
        self.time += dt

        # Wavelength random walk
        wavelength_shift = self.rng.normal(0, self.noise * np.sqrt(dt))

        drift = self.sensitivity * wavelength_shift * self.mzi_sensitivity
        self.cumulative_drift += drift
        return drift


# =============================================================================
# 6. ELECTRONIC DRIFT MODELS
# =============================================================================

class DACDrift(DriftModel):
    """
    6.1 DAC Drift

    Characteristics:
    - Timescale: Hours (temperature-related)
    - Spatial pattern: Per-channel (random)
    - Magnitude: LSB-level drifts
    - Compressibility: LOW (independent channels)

    Mitigation:
    - High-quality DACs
    - Periodic calibration
    - Temperature-stable references
    """

    def __init__(self, n_mzis, dac_noise=0.0001, rng=None):
        super().__init__(n_mzis, rng)
        self.dac_noise = dac_noise

    def step(self, dt=1.0):
        self.time += dt

        # Independent drift per channel
        drift = self.rng.normal(0, self.dac_noise * np.sqrt(dt), self.n_mzis)
        self.cumulative_drift += drift
        return drift


# =============================================================================
# COMPREHENSIVE DRIFT SUMMARY
# =============================================================================

def print_drift_summary():
    """Print comprehensive drift classification."""
    print(DRIFT_TAXONOMY)

    summary = """
┌───────────────────────────────────────────────────────────────────────────────┐
│                           DRIFT CHARACTERISTICS SUMMARY                        │
├────────────────────┬──────────────┬──────────────┬──────────────┬─────────────┤
│ Drift Type         │ Timescale    │ Compress.    │ Reversible?  │ Calibration │
├────────────────────┼──────────────┼──────────────┼──────────────┼─────────────┤
│ Ambient thermal    │ min-hours    │ HIGH         │ Yes          │ ~10 min     │
│ Self-heating       │ ms-sec       │ MEDIUM       │ Yes          │ ~1 sec      │
│ Thermal crosstalk  │ sec          │ HIGH         │ Yes          │ ~10 sec     │
│ Vibration          │ ms           │ HIGH         │ Yes          │ ~1 ms       │
│ Stress relaxation  │ hours-days   │ MEDIUM       │ No           │ ~1 hour     │
│ Densification      │ months       │ VERY HIGH    │ No           │ ~1 week     │
│ Electromigration   │ months       │ LOW          │ No           │ ~1 week     │
│ Humidity           │ hours        │ HIGH         │ Yes          │ ~1 hour     │
│ Wavelength         │ sec-min      │ HIGH         │ Yes          │ ~10 sec     │
│ DAC drift          │ hours        │ LOW          │ Yes          │ ~1 hour     │
└────────────────────┴──────────────┴──────────────┴──────────────┴─────────────┘

CALIBRATION STRATEGY BY DRIFT TYPE:
───────────────────────────────────

┌─────────────────────────────────────────────────────────────────────────────┐
│ HIGH COMPRESSIBILITY (Smooth/Structured)                                    │
│ → Use COMPRESSIVE SENSING (4-10x measurement reduction)                     │
│                                                                             │
│   • Ambient thermal: Smooth gradient                                        │
│   • Wavelength drift: Systematic pattern                                    │
│   • Vibration: Few mechanical modes                                         │
│   • Humidity: Edge-to-center gradient                                       │
│   • Densification: Uniform shift                                            │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ LOW COMPRESSIBILITY (Random/Independent)                                    │
│ → Use FULL MEASUREMENT or per-MZI calibration                              │
│                                                                             │
│   • DAC drift: Independent channels                                         │
│   • Electromigration: Usage-dependent, random                              │
│   • Fabrication errors: Random (initial only)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ FAST DRIFT (< 1 second)                                                     │
│ → Use FEEDBACK CONTROL (not batch calibration)                             │
│                                                                             │
│   • Vibration: kHz feedback loop                                            │
│   • Self-heating: Power-aware compensation                                  │
│   • Mode hopping: Fast wavelength lock                                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ IRREVERSIBLE DRIFT (Aging)                                                  │
│ → Use LIFETIME MODELS + periodic full recalibration                        │
│                                                                             │
│   • Stress relaxation: Burn-in + monthly checks                            │
│   • Densification: Annual recalibration                                     │
│   • Electromigration: Usage tracking + prediction                          │
└─────────────────────────────────────────────────────────────────────────────┘
"""
    print(summary)


def simulate_combined_drift(n_modes=16, n_timesteps=1000):
    """
    Simulate realistic combined drift from multiple sources.
    """
    print()
    print("=" * 75)
    print("COMBINED DRIFT SIMULATION")
    print("=" * 75)
    print()

    mesh = ClementsMesh(n_modes)
    n_mzis = mesh.n_mzis

    print(f"System: {n_modes} modes, {n_mzis} MZIs")
    print(f"Simulating {n_timesteps} timesteps")
    print()

    # Create all drift models
    drift_models = {
        'Ambient Thermal': AmbientThermalDrift(n_mzis),
        'Self-Heating': SelfHeatingDrift(n_mzis),
        'Thermal Crosstalk': ThermalCrosstalkDrift(n_mzis),
        'Vibration': VibrationDrift(n_mzis),
        'Wavelength': WavelengthDrift(n_mzis),
        'DAC': DACDrift(n_mzis),
    }

    # Setup target
    rng = np.random.default_rng(42)
    thetas = rng.uniform(0, np.pi/2, n_mzis)
    phis = rng.uniform(0, 2*np.pi, n_mzis)
    mesh.set_phases(thetas, phis)
    U_target = mesh.unitary(include_noise=False)

    # Track contributions
    contributions = {name: [] for name in drift_models}
    total_drift_history = []
    fidelity_history = []

    for t in range(n_timesteps):
        total_drift = np.zeros(n_mzis)

        for name, model in drift_models.items():
            drift = model.step(dt=1.0)
            total_drift += drift
            contributions[name].append(np.std(model.cumulative_drift))

        total_drift_history.append(np.std(total_drift))

        # Measure fidelity
        mesh.apply_noise(total_drift, np.zeros(n_mzis))
        U_drifted = mesh.unitary(include_noise=True)
        fid = fidelity(U_target, U_drifted)
        fidelity_history.append(fid)
        mesh.clear_noise()

    # Analyze contributions
    print("Drift Contributions (std dev at final timestep):")
    print("-" * 50)

    final_contributions = {name: contributions[name][-1] for name in drift_models}
    total_var = sum(c**2 for c in final_contributions.values())

    for name, std in sorted(final_contributions.items(), key=lambda x: -x[1]):
        pct = (std**2 / total_var) * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<20}: {std:.4f} rad ({pct:5.1f}%) {bar}")

    print()
    print(f"Total drift std: {np.sqrt(total_var):.4f} rad")
    print(f"Final fidelity: {fidelity_history[-1]:.6f}")
    print()

    # Compressibility analysis
    print("Compressibility of Combined Drift:")
    print("-" * 50)

    combined_drift = np.zeros(n_mzis)
    for model in drift_models.values():
        combined_drift += model.cumulative_drift

    # Fourier analysis
    fft = np.fft.fft(combined_drift)
    power = np.abs(fft)**2
    sorted_power = np.sort(power)[::-1]
    total_power = np.sum(power)

    for k in [5, 10, 20, 50]:
        energy = np.sum(sorted_power[:k]) / total_power * 100
        print(f"  Top {k} modes: {energy:.1f}% of energy")

    return contributions, fidelity_history


def main():
    print()
    print("╔" + "═"*73 + "╗")
    print("║" + " "*18 + "DRIFT CLASSIFICATION FOR PHOTONICS" + " "*20 + "║")
    print("╚" + "═"*73 + "╝")

    # Print taxonomy
    print_drift_summary()

    # Simulate combined drift
    contributions, fidelities = simulate_combined_drift(n_modes=16, n_timesteps=1000)

    # Recommendations
    print()
    print("=" * 75)
    print("RECOMMENDED CALIBRATION HIERARCHY")
    print("=" * 75)
    print()

    hierarchy = """
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CALIBRATION HIERARCHY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LEVEL 0: Fabrication (one-time)                                           │
│  └── Full Jacobian calibration, Bayesian estimation                        │
│      Frequency: Once at chip bring-up                                       │
│      Method: Full measurement (no compression)                              │
│                                                                             │
│  LEVEL 1: Thermal stabilization (continuous)                               │
│  └── TEC feedback loop, temperature monitoring                             │
│      Frequency: 100 Hz feedback loop                                        │
│      Method: Direct temperature sensing                                     │
│                                                                             │
│  LEVEL 2: Fast drift tracking (1-10 Hz)                                    │
│  └── Compressive sensing for thermal + wavelength drift                    │
│      Frequency: Every 0.1-1 seconds                                        │
│      Method: 4-10x compressed measurements                                  │
│                                                                             │
│  LEVEL 3: Slow drift tracking (0.01 Hz)                                    │
│  └── Full recalibration for accumulated errors                             │
│      Frequency: Every 10-60 minutes                                        │
│      Method: Full measurement with Bayesian update                         │
│                                                                             │
│  LEVEL 4: Aging compensation (weekly/monthly)                              │
│  └── Lifetime model update, heater recalibration                           │
│      Frequency: Weekly to monthly                                          │
│      Method: Full characterization                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

EXAMPLE TIMELINE FOR 10,000 MZI SYSTEM:
───────────────────────────────────────

  Time        │ Action                          │ Duration
  ────────────┼─────────────────────────────────┼──────────
  t=0         │ Initial full calibration        │ 20 min
  t=0.1s      │ Compressed drift update #1      │ 3 sec
  t=0.2s      │ Compressed drift update #2      │ 3 sec
  ...         │ (continue every 0.1s)           │ ...
  t=10min     │ Full recalibration              │ 5 min
  t=20min     │ Full recalibration              │ 5 min
  ...         │ (continue every 10 min)         │ ...
  t=1 week    │ Aging model update              │ 30 min
  ...         │ (continue weekly)               │ ...
"""
    print(hierarchy)


if __name__ == "__main__":
    main()
