# Differentiable Voronoi Lattice Optimization

A differentiable Voronoi-based lattice generator for shell structures optimized using **neural networks** and **finite element methods (FEM)**.

The system learns to place Voronoi seeds and control strut thickness so the resulting structure:

* satisfies a **target volume fraction**
* minimizes **structural compliance**
* forms clear **Voronoi struts**
* remains **fully differentiable for gradient-based optimization**

The framework is implemented with **PyTorch**, allowing gradients to propagate through:

```
Seeds → Voronoi Geometry → Density Field → FEM → Compliance Loss
```

---

# Project Architecture

The system contains four main components.

## 1. PPNet (Neural Parameter Predictor)

PPNet predicts geometric parameters from a latent context vector.

### Outputs

| Parameter     | Shape    | Description                          |
| ------------- | -------- | ------------------------------------ |
| seeds_raw     | (S × 2)  | Voronoi seed coordinates in UV space |
| w_raw         | scalar   | global strut half-width              |
| h_raw         | scalar   | optional height parameter            |
| gate_probs    | (S)      | optional seed gating                 |
| theta / a_raw | optional | anisotropy parameters                |

### Seed Refinement

Seeds are refined using a **logit-space offset parameterization**:

```
seeds = sigmoid(logit(uv_init) + tanh(delta))
```

This avoids gradient problems caused by clamping.

---

# 2. Voronoi Decoder

The decoder converts predicted parameters into a **density field over the surface mesh**.

### Pipeline

```
Seeds → Voronoi diagram → bisector distance → density field
```

The key geometric quantity:

```
d(x) = signed distance to Voronoi bisector
```

Density is defined as:

```
rho(x) = sigmoid((w - |d(x)|) / beta)
```

Where

* `w` = strut half-width
* `beta` = smoothing parameter

This produces a **smooth differentiable strut band centered on Voronoi edges**.

---

# 3. FEM Solver

The density field is mapped to FEM material density.

The solver returns:

* structural **compliance**
* **stress field**

Compliance is used as the main structural objective.

To avoid numerical instability:

```
rho_FEM = max(rho, density_floor)
```

---

# 4. Training Loop (NN_Trainer)

The trainer optimizes PPNet parameters using multiple losses.

Optimized variables include:

* seed positions
* strut width
* optional anisotropy parameters

---

# Loss Functions

The total loss combines multiple objectives:

```
L_total =
  λ_vol   L_vol
+ λ_fem   L_fem
+ λ_rep   L_seed_repulsion
+ λ_bnd   L_boundary
+ λ_strut L_strut
```

---

## Volume Loss

Matches the target volume fraction.

```
vol = Σ (rho * vertex_area) / Σ(vertex_area)
L_vol = (vol - target_volfrac)^2
```

---

## FEM Compliance Loss

Minimizes structural compliance.

```
L_fem = compliance / normalization_factor
```

---

## Seed Repulsion Loss

Prevents Voronoi seeds from collapsing together.

```
L_rep = mean(exp(-||si - sj||² / σ²))
```

---

## Boundary Repulsion Loss

Prevents seeds from approaching the surface boundary.

```
penalty = exp(-distance_to_boundary / margin)
```

---

# Strutness Loss

Ensures material aligns with Voronoi edges.

It contains two components.

### Edge Solidification

Encourages density near Voronoi bisectors.

```
Lse = mean((1 - rho_edge)^2)
```

### Void Suppression

Encourages empty space away from edges.

```
Lsv = mean(rho_void^2)
```

### Combined Loss

```
L_strut = λ_se Lse + λ_sv Lsv
```

This enforces:

* solid struts
* clear void regions

---

# Geometric Width Parameter

Previously strut thickness depended on a Voronoi threshold parameter `τ`, which caused instability.

The updated model introduces a **direct geometric width parameter**:

```
w = strut half-width
thickness = 2w
```

The neural network predicts:

```
w_raw → w
```

This provides:

* explicit thickness control
* smooth differentiability
* stable optimization

---

# Training Diagnostics

During training the system logs:

* `L_total`
* `L_vol`
* `L_fem`
* `L_strut`
* `L_rep`
* `L_bnd`
* volume fraction
* compliance
* `w_geo` (average width)
* `Lse`
* `Lsv`
* `rho(min/mean/max)`
* `Δrho`
* `Δseed`
* gradient magnitude

These diagnostics help monitor optimization behaviour.

---

# Visualization

Two visualization modes are available.

### Stepwise Evolution

Displays:

* initial density
* mid optimization
* final density

along with seed positions.

### Final Structure

Optionally shows a **thresholded density visualization**.

---

# Current System Behaviour

Typical observations:

* the optimizer strongly adjusts **strut width**
* width decreases to reduce volume
* seeds move moderately
* topology remains mostly fixed

Typical evolution:

```
large width → high volume
optimizer shrinks width → volume decreases
structure converges
```

---

# Current Challenges

### Volume vs Structural Strength

Very low volume fractions force struts to become extremely thin.

### Strut Solidification

Struts can become weak if width shrinks excessively.

### Topology Change

Seed motion is limited and topology changes are rare.

---

# Future Work

Potential improvements include:

* adaptive seed insertion/removal
* topology exploration strategies
* anisotropic Voronoi structures
* multi-objective structural optimization

---
