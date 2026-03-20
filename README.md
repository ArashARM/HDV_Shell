# Differentiable Voronoi Lattice Optimization — Project Summary

## 1. Purpose

This project aims to **automatically generate optimized lattice structures on 3D surfaces** using a fully differentiable pipeline.

The key idea is to:

* Represent lattice geometry using **Voronoi diagrams in 2D surface parameter space (UV)**
* Map this structure to **3D density and fiber fields**
* Optimize both **geometry (seed positions)** and **material distribution (strut thickness)**

The optimization objective typically includes:

* Target volume fraction
* Structural performance (e.g., compliance via FEM)
* Geometric regularity (repulsion, boundary behavior, strut structure)

---

## 2. Core Idea

Instead of explicitly designing lattices, the system **learns them** by:

* Predicting **Voronoi seed positions**
* Controlling **strut thickness and anisotropy**
* Evaluating performance using **differentiable physics (FEM)**

Everything is optimized **end-to-end via gradient descent**.

---

## 3. High-Level Pipeline

```
Latent Context
    ↓
PPNet (Neural Network)
    ↓
Voronoi Seeds in UV Space
    ↓
Voronoi Decoder
    ↓
3D Mapping (Geometry → Density + Fiber Directions)
    ↓
FEM Solver (Physics Evaluation)
    ↓
Loss Computation
    ↓
Backpropagation (Update Neural Parameters)
```

---

## 4. Main Components

### 4.1 PPNet (Parameter Prediction Network)

* Inputs:

  * Latent context vector
  * Initial seed positions (UV)
* Outputs:

  * Refined Voronoi seeds
  * Strut width (and optionally height)
  * Optional anisotropy parameters

---

### 4.2 Voronoi Decoder

* Takes UV seeds and surface information
* Builds a **soft Voronoi diagram**
* Produces:

  * Density field on the surface
  * Fiber directions aligned with Voronoi edges
  * Edge-field information for structural regularization

---

### 4.3 Surface Mapping

* Converts UV-based results to 3D using:

  * Surface derivatives (Xu, Xv)
* Outputs:

  * 3D density distribution
  * 3D fiber orientation field

---

### 4.4 FEM Module

* Uses density and fiber fields
* Computes structural response (e.g., compliance)
* Fully differentiable → gradients flow back to seeds and network

---

### 4.5 Loss Functions

Typical terms include:

* **Volume loss**

  * Enforces target material usage

* **Repulsion loss**

  * Prevents seed collapse

* **Boundary loss**

  * Keeps seeds away from edges

* **Strut loss**

  * Encourages meaningful lattice structure

* **FEM loss**

  * Optimizes mechanical performance

---

## 5. Training Loop Overview

For each iteration:

1. Predict seeds and parameters using PPNet
2. Decode Voronoi structure per face
3. Assemble global fields (density, fiber)
4. Compute losses (volume + geometry + FEM)
5. Backpropagate gradients
6. Update network parameters

---

## 6. Key Design Principles

* **Work in UV space** for simplicity and consistency across surfaces
* **Map to 3D only when needed** for physics evaluation
* **Keep everything differentiable**
* **Jointly optimize geometry and physics**
* **Use neural networks as parametric generators, not black boxes**

---

## 7. Practical Notes

* Seed updates must be **bounded and stable**
* FEM gradients can be **unstable and should be introduced gradually**
* Multi-face training requires:

  * Proper normalization
  * Careful gradient scaling
* Debugging should isolate:

  * Geometry path
  * FEM path

---

## 8. End Goal

A system that can:

* Automatically generate lattice structures
* Adapt to arbitrary surfaces
* Optimize both shape and mechanical behavior
* Remain fully differentiable and trainable

---

## 9. Summary

This project combines:

* Geometry (Voronoi diagrams)
* Neural networks (PPNet)
* Physics (FEM)

into a single differentiable framework that enables **learning-based structural design on surfaces**.
