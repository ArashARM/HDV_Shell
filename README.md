# 🧠 Differentiable Voronoi Lattice Optimization

\[![Python](https://img.shields.io/badge/python-3.10-blue.svg)\]
\[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)\]
\[![License](https://img.shields.io/badge/license-MIT-green.svg)\]

A differentiable framework for generating and optimizing lattice
structures on surfaces using **Voronoi diagrams**, **neural networks**,
and **finite element methods (FEM)**.

------------------------------------------------------------------------

## 🚀 TL;DR

We learn optimized lattice structures by jointly optimizing:

-   **Voronoi seed positions** (geometry)
-   **Strut thickness** (material distribution)

This is achieved via **end-to-end differentiable FEM**, where Voronoi
diagrams are computed in **2D surface parameter space (u,v)** and mapped
to **3D geometry, density, and fiber directions**.

------------------------------------------------------------------------

## ⭐ Key Contributions

-   🔷 **UV-Space Voronoi Representation**\
    Voronoi diagrams are constructed in the **2D parametric domain
    (u,v)**.

-   🌐 **Consistent Mapping to 3D**\
    Produces:

    -   Density field `ρ(x)`
    -   Fiber directions aligned with Voronoi edges

-   🔁 **Fully Differentiable Pipeline**\
    Gradients propagate through: Seeds → Geometry → Density → FEM → Loss

-   🧠 **Neural Parameterization (PPNet)**\
    Predicts seed locations, strut width, and anisotropy

-   🏗️ **Coupled Geometry + Mechanics Optimization**\
    Optimizes compliance and geometry jointly

------------------------------------------------------------------------

## 🔁 End-to-End Pipeline

Latent Context → PPNet → Voronoi (UV) → 3D Mapping → Density + Fibers →
FEM → Loss → Backprop

------------------------------------------------------------------------

## 📌 Conclusion

Voronoi diagrams in surface parameter space enable **differentiable
lattice generation in 3D**, allowing joint optimization of geometry and
mechanics via neural networks and FEM.
