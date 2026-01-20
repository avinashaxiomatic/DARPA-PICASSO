# Experimental Protocols for Large-Scale Photonic Systems

## PICASSO: From Simulation to Hardware

---

## 1. Target Applications for 12,000+ MZI Systems

### 1.1 Optical Neural Network Accelerators

**Scale**: 155×155 unitary matrix (11,935 MZIs)

**Application**: Deep learning inference acceleration
- Matrix-vector multiplication at speed of light
- Energy efficiency: ~1 fJ/MAC vs ~1 pJ/MAC for electronic
- Latency: <1 ns per layer vs ~100 ns for GPU

**Why 12K MZIs?**
- 155-dimensional input vectors match common embedding sizes
- Transformer attention heads (64-256 dimensions)
- Convolutional filter banks

**Calibration Requirements**:
- Initial calibration: <30 minutes acceptable
- Drift correction: Every ~10 minutes (thermal)
- Target fidelity: >99% for accurate inference

---

### 1.2 Quantum Photonic Processors

**Scale**: 155-mode linear optical network

**Application**: Boson sampling, Gaussian boson sampling
- Computational advantage demonstrations
- Molecular vibronic spectra simulation
- Graph optimization problems

**Why 12K MZIs?**
- Boson sampling with 50+ photons requires >100 modes
- Gaussian boson sampling scales with mode count
- Quantum advantage threshold: ~50-100 photons in ~100+ modes

**Calibration Requirements**:
- Extremely high fidelity: >99.9% per MZI
- Must maintain quantum coherence during calibration
- Non-destructive calibration (can't measure quantum states directly)

---

### 1.3 Optical MIMO & Beamforming

**Scale**: 155-antenna array processing

**Application**: 5G/6G wireless, radar, LIDAR
- Massive MIMO precoding
- Adaptive beamforming
- Direction of arrival estimation

**Why 12K MZIs?**
- 5G massive MIMO: 64-256 antennas typical
- Automotive radar: 100+ virtual channels
- Real-time reconfiguration required

**Calibration Requirements**:
- Fast recalibration: <1 second
- Must calibrate during operation (in-situ)
- Robust to temperature variations

---

### 1.4 Optical Reservoir Computing

**Scale**: 155-node reservoir

**Application**: Time-series prediction, chaotic systems
- Weather forecasting
- Financial modeling
- Speech recognition

**Why 12K MZIs?**
- Reservoir performance scales with size
- 155 nodes competitive with state-of-art electronic reservoirs
- Inherent nonlinearity from optical interference

---

## 2. Experimental Calibration Protocols

### 2.1 Protocol A: Full Jacobian Calibration (Gold Standard)

**When to use**: Initial system bring-up, periodic recalibration

**Procedure**:
```
1. SET all phases to known reference (θ=0, φ=0)
2. MEASURE reference output matrix U_ref
3. FOR each target unitary U_target:
   a. COMPUTE ideal phases via decomposition
   b. SET phases to ideal values
   c. MEASURE actual output U_actual
   d. COMPUTE deviation ΔU = U_actual - U_target
   e. COMPUTE Jacobian J = ∂U/∂θ (numerically or analytically)
   f. SOLVE regularized system: δθ = (J^T J + λI)^{-1} J^T ΔU
   g. APPLY correction: θ_corrected = θ_ideal - α·δθ
   h. VERIFY fidelity meets threshold
4. STORE calibration parameters
```

**Resources Required**:
- Coherent light source (tunable laser)
- N² photodetectors (or scanning single detector)
- Phase modulators with <0.01 rad precision
- Computation: O(N⁴) for Jacobian, O(N³) for solve

**Time Estimate** (155 modes):
- Jacobian measurement: ~10 minutes
- Solve: ~20 minutes
- Verification: ~5 minutes
- **Total: ~35 minutes**

---

### 2.2 Protocol B: Incremental Bayesian Update (Operational)

**When to use**: Drift correction during operation, thermal compensation

**Procedure**:
```
1. INITIALIZE prior from last calibration: θ_prior, Σ_prior
2. DURING operation:
   a. PERIODICALLY measure subset of outputs (sparse sampling)
   b. COMPUTE expected vs actual for sampled outputs
   c. UPDATE posterior using Bayesian inference:
      θ_posterior = θ_prior + K·(y_measured - y_expected)
      where K = Σ_prior·H^T·(H·Σ_prior·H^T + R)^{-1}
   d. APPLY small corrections incrementally
3. IF drift exceeds threshold:
   TRIGGER full recalibration (Protocol A)
```

**Resources Required**:
- Subset of photodetectors (√N sufficient)
- Background computation capability
- Phase modulator fine adjustment

**Time Estimate**:
- Per update: <1 second
- Can run continuously in background

---

### 2.3 Protocol C: Hierarchical Block Calibration (Scalable)

**When to use**: Very large systems (>20K MZIs), modular architectures

**Procedure**:
```
1. PARTITION mesh into blocks of ~1000 MZIs each
2. FOR each block independently:
   a. ISOLATE block (set surrounding phases to pass-through)
   b. CALIBRATE block using Protocol A
   c. STORE block calibration
3. CALIBRATE inter-block coupling:
   a. ACTIVATE adjacent block pairs
   b. MEASURE cross-coupling matrix
   c. CORRECT for inter-block phase errors
4. VERIFY global fidelity
```

**Advantages**:
- Memory efficient: O(N_block²) instead of O(N²)
- Parallelizable across blocks
- Fault isolation (identify defective blocks)

**Time Estimate** (12K MZIs = 12 blocks):
- Per block: ~3 minutes
- Inter-block: ~10 minutes
- **Total: ~45 minutes** (with parallelization: ~15 minutes)

---

### 2.4 Protocol D: Training-Based Calibration (ML-Enhanced)

**When to use**: Repeated similar operations, neural network weights

**Procedure**:
```
1. DEFINE target operation set {U_1, U_2, ..., U_K}
2. FOR each target:
   a. START with decomposition-based phases
   b. MEASURE actual output
   c. BACKPROPAGATE error through differentiable model
   d. UPDATE phases using gradient descent
   e. REPEAT until convergence
3. LEARN systematic error model:
   a. FIT neural network: θ_correction = f_NN(θ_ideal, T, λ, ...)
   b. INCLUDE environmental parameters (temperature, wavelength)
4. DEPLOY learned correction model
```

**Advantages**:
- Amortized calibration cost over many operations
- Learns systematic fabrication errors
- Can incorporate environmental sensing

---

## 3. Hardware Requirements

### 3.1 Measurement System

| Component | Specification | Purpose |
|-----------|--------------|---------|
| Laser | 1550nm, <100kHz linewidth | Coherent illumination |
| Photodetector array | 155×155 InGaAs, >1GHz BW | Output measurement |
| DAC array | 16-bit, 12K channels | Phase control |
| ADC array | 14-bit, 24K channels | Detector readout |
| FPGA | Xilinx VU13P or equiv | Real-time processing |

### 3.2 Environmental Control

| Parameter | Requirement | Reason |
|-----------|-------------|--------|
| Temperature | ±0.01°C stability | Phase drift ~0.1 rad/°C |
| Vibration | <10nm RMS | Path length stability |
| Humidity | <5% variation | Cladding index change |

### 3.3 Computation

| Task | Hardware | Time (12K MZIs) |
|------|----------|-----------------|
| Jacobian storage | 2.3 GB RAM | - |
| SVD decomposition | 32-core CPU | ~5 min |
| Tikhonov solve | GPU (RTX 4090) | ~30 sec |
| Real-time update | FPGA | <1 ms |

---

## 4. Calibration Performance Targets

### 4.1 Fidelity Targets by Application

| Application | Target Fidelity | Acceptable Drift |
|-------------|-----------------|------------------|
| Neural network inference | >99.0% | 0.5%/hour |
| Quantum computing | >99.9% | 0.01%/hour |
| MIMO beamforming | >98.0% | 1%/minute |
| Reservoir computing | >95.0% | 2%/hour |

### 4.2 Calibration Time Budget

| Phase | Time | Frequency |
|-------|------|-----------|
| Initial bring-up | 30 min | Once |
| Thermal equilibration | 60 min | Once per power-on |
| Periodic recalibration | 5 min | Every 4 hours |
| Drift correction | 1 sec | Continuous |

---

## 5. Experimental Validation Plan

### Phase 1: Small-Scale Validation (Months 1-6)
- **Scale**: 16 modes (120 MZIs)
- **Goal**: Validate protocols on existing hardware
- **Deliverables**:
  - Protocol A demonstrated with >95% recovery
  - Protocol B real-time updates working
  - Thermal drift characterization

### Phase 2: Medium-Scale Demonstration (Months 7-12)
- **Scale**: 64 modes (2,016 MZIs)
- **Goal**: Demonstrate scaling with custom chip
- **Deliverables**:
  - Full Bayesian calibration in <5 minutes
  - Block calibration validated
  - Neural network inference demo

### Phase 3: Large-Scale System (Months 13-24)
- **Scale**: 155 modes (11,935 MZIs)
- **Goal**: Full system with application demos
- **Deliverables**:
  - 12K MZI calibration in <30 minutes
  - Quantum advantage experiment
  - Optical neural network benchmark vs GPU

---

## 6. Risk Mitigation

### 6.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Fabrication variation >1° | Medium | High | Bayesian calibration handles up to 3° |
| Thermal drift faster than expected | Medium | Medium | Continuous Protocol B updates |
| Detector noise limiting accuracy | Low | High | Averaging, lock-in detection |
| Memory limits for Jacobian | Low | Medium | Block calibration (Protocol C) |

### 6.2 Fallback Strategies

1. **If full Jacobian fails**: Use block calibration
2. **If real-time update too slow**: Pre-compute correction tables
3. **If fidelity insufficient**: Iterate calibration multiple rounds

---

## 7. Metrics and Success Criteria

### 7.1 Primary Metrics

1. **Fidelity Recovery**: (F_calibrated - F_raw) / (1 - F_raw) > 90%
2. **Calibration Time**: < 30 minutes for 12K MZIs
3. **Drift Tolerance**: Maintain >95% fidelity for >1 hour
4. **Update Latency**: < 1 second for incremental correction

### 7.2 Stretch Goals

1. **Real-time reconfiguration**: Change unitary in <1ms
2. **Closed-loop fidelity**: Maintain >99% continuously
3. **Scale to 50K MZIs**: With hierarchical calibration

---

*Document Version 1.0*
*PICASSO Project - January 2026*
