# 🧠 Differentiable Voronoi Lattice Optimization

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)]
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]
[![License](https://img.shields.io/badge/license-MIT-green.svg)]

A differentiable Voronoi-based lattice generator for shell structures optimized using **neural networks** and **finite element methods (FEM)**.

The system learns to place Voronoi seeds and control strut thickness so the resulting structure:

- satisfies a **target volume fraction**
- minimizes **structural compliance**
- forms clear **Voronoi struts**
- remains **fully differentiable for gradient-based optimization**

---

## 🔁 Differentiable Pipeline


Latent Context
↓
PPNet (Neural Predictor)
↓
Voronoi Decoder (Differentiable Geometry)
↓
Density Field ρ(x)
↓
FEM Solver
↓
Compliance Loss
↓
Backpropagation


Gradients propagate through the full chain:


Seeds → Geometry → Density → FEM → Loss


---

## 🏗️ Project Architecture

The system consists of four main components.

---

### 1. 🧠 PPNet (Neural Parameter Predictor)

Predicts geometric parameters from a latent vector.

| Parameter     | Shape    | Description                          |
|--------------|----------|--------------------------------------|
| seeds_raw    | (S × 2)  | Voronoi seed coordinates (UV space)  |
| w_raw        | scalar   | strut half-width                     |
| h_raw        | scalar   | optional height                      |
| gate_probs   | (S)      | optional seed gating                 |
| theta / a_raw| optional | anisotropy parameters                |

#### Seed Refinement


seeds = sigmoid(logit(uv_init) + tanh(delta))


✔ stable gradients  
✔ no hard clipping  
✔ bounded domain  

---

### 2. 🔷 Voronoi Decoder (Differentiable Geometry)

Converts predicted parameters into a **density field over the surface mesh**.

#### Pipeline


Seeds → Soft Voronoi → Bisector Distance → Density Field


#### Density Definition


ρ(x) = sigmoid((w - |d(x)|) / beta)


Where:

- `d(x)` = distance to Voronoi bisector  
- `w` = strut half-width  
- `beta` = smoothing parameter  

---

### 🔶 Differentiable Edge Field

To capture Voronoi edges:


edge_field = 1 - ∏ (1 - band_ij)


This:

- approximates the union of Voronoi edges  
- remains fully differentiable  
- is critical for clean strut formation  

---

### 🔷 Boundary Handling

To prevent boundary artifacts:


weight = 1 - rho_boundary + alpha * rho_boundary


Used in volume computation to reduce boundary material.

---

### 3. 🏗️ FEM Solver

Maps density to structural response.

Outputs:

- compliance
- optional stress field

#### Stability


rho_FEM = max(rho, density_floor)


---

### 4. 🔁 Training Loop (NN_Trainer)

Optimizes:

- seed positions
- strut width
- optional anisotropy

---

## 🎯 Loss Function


L_total =
λ_vol L_vol

λ_fem L_fem

λ_rep L_rep

λ_bnd L_bnd

λ_strut L_strut


---

### 📦 Volume Loss


vol = Σ (ρ A_v) / Σ(A_v)
L_vol = (vol - target)^2


---

### 🏗️ FEM Loss


L_fem = compliance / normalization_factor


---

### 🔁 Seed Repulsion


L_rep = mean(exp(-||si - sj||² / σ²))


---

### 🚧 Boundary Repulsion


penalty = exp(-distance_to_boundary / margin)


---

### 🔷 Strutness Loss

#### Edge solidification


L_edge = mean(edge_field * (1 - ρ)^2)


#### Void suppression


L_void = mean((1 - edge_field) * ρ^2)


#### Combined


L_strut = λ_edge L_edge + λ_void L_void


---

## 📏 Geometric Width Parameter


w = strut half-width
thickness = 2w


✔ explicit  
✔ differentiable  
✔ stable  

---

## 🎨 Visualization

### Stepwise Evolution

| Initial | Mid | Final |
|--------|-----|------|
| ![](assets/init.png) | ![](assets/mid.png) | ![](assets/final.png) |

---

### Final Structure


ρ > threshold → solid


Recommended:

- `0.5` → true geometry  
- `0.6–0.7` → cleaner struts  

---

## 🎥 Evolution (GIF)

<p align="center">
  <img src="assets/stepwise.gif" width="80%">
</p>

---

## ⚙️ Recommended Settings


beta = 0.005
lam_strut = 0.1 – 0.2
lam_fem = 3 – 5


---

## 📊 Training Diagnostics

Logged per iteration:

- total loss + components
- compliance
- volume fraction
- density stats (min / mean / max)
- Δrho, Δseed
- gradient magnitude
- width (`w_geo`)

---

## ⚠️ Common Issues

| Problem | Fix |
|--------|-----|
| blurry structures | decrease `beta` |
| weak struts | increase `lam_strut` |
| FEM dominates | reduce `lam_fem` |
| boundary artifacts | enable boundary weighting |

---

## 🧠 Key Insights

- structure emerges from **geometry + optimization**
- width is the **main control variable**
- FEM provides **global guidance**
- strutness enforces **local structure**

---

# 📊 Results (TO FILL)

## Quantitative Results

| Metric | Value |
|-------|------|
| Target Volume Fraction | XXX |
| Achieved Volume Fraction | XXX |
| Compliance | XXX |
| Mean Strut Width | XXX |

---

## Qualitative Observations

- [ ] Struts are continuous and well-defined  
- [ ] No boundary artifacts  
- [ ] Clear Voronoi topology  
- [ ] Good structural performance  

---

## Comparison (Optional)

| Method | Compliance ↓ | Volume | Notes |
|-------|-------------|--------|------|
| Ours | XXX | XXX | Voronoi structured |
| Baseline (SIMP) | XXX | XXX | irregular |

---

# 🔮 Future Work

- adaptive seed insertion/removal  
- topology exploration  
- anisotropic Voronoi lattices  
- manufacturing constraints  
- binary (threshold-free) output  

---

# 📄 Paper (Work in Progress)

SIGGRAPH-style paper draft available in `/paper/`.

---

# ⭐ Acknowledgements

If you use this work, please consider citing (to be added).

---

# 📌 Summary

This framework enables:

- differentiable Voronoi geometry  
- neural topology optimization  
- FEM-driven structural design  
- interpretable lattice structures  
