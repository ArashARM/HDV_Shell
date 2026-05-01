# HDV Shell: Differentiable Voronoi Lattice Optimization

## Purpose

This project aims to **automatically generate optimized lattice structures on 3D surfaces** using a fully differentiable pipeline.

The key idea is to:

* Represent lattice geometry using **Voronoi diagrams in 2D surface parameter space (UV)**
* Map this structure to **3D density and fiber fields**
* Optimize both **geometry (seed positions)** and **material distribution (strut thickness)**

The optimization objective typically includes:

* Target volume fraction
* Structural performance (e.g., compliance via FEM)
* Geometric regularity (repulsion, boundary behavior, strut structure)

## Core Idea

Instead of explicitly designing lattices, the system **learns them** by:

* Predicting **Voronoi seed positions**
* Controlling **strut thickness and anisotropy**
* Evaluating performance using **differentiable physics (FEM)**

Everything is optimized **end-to-end via gradient descent**.

## High-Level Pipeline

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

## Main Components

### PPNet

* Inputs:

  * Latent context vector
  * Initial seed positions (UV)
* Outputs:

  * Refined Voronoi seeds
  * Strut width (and optionally height)
  * Optional anisotropy parameters

### Voronoi Decoder

* Takes UV seeds and surface information
* Builds a **soft Voronoi diagram**
* Produces:

  * Density field on the surface
  * Fiber directions aligned with Voronoi edges
  * Edge-field information for structural regularization

### Surface Mapping

* Converts UV-based results to 3D using:

  * Surface derivatives (Xu, Xv)
* Outputs:

  * 3D density distribution
  * 3D fiber orientation field

### FEM Module

* Uses density and fiber fields
* Computes structural response (e.g., compliance)
* Fully differentiable → gradients flow back to seeds and network

### Loss Functions

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

## Training Loop Overview

For each iteration:

1. Predict seeds and parameters using PPNet
2. Decode Voronoi structure per face
3. Assemble global fields (density, fiber)
4. Compute losses (volume + geometry + FEM)
5. Backpropagate gradients
6. Update network parameters

## Notebook Workflow

The main workflow is driven from `Main.ipynb`.

Typical setup:

```python
loading_img = shell_problem.show_voxels_surface_and_bc(
    return_img=True,
    off_screen=True,
    window_size=(560, 430),
    show=True,
)
```

`show=True` opens a larger interactive visualization, while `return_img=True` also returns the rendered loading/boundary-condition image for the timelapse frames.

Then pass this image into the trainer:

```python
trainer = NN_Trainer(
    generator=generator,
    viz=viz,
    fem=fem,
    shell_problem=shell_problem,
    config=cfg,
    loading_img=loading_img,
)
```

If `loading_img` is not provided, the trainer will try to generate one automatically from `shell_problem`.

## Timelapse Outputs

Timelapse generation is controlled by `TrainingConfig`:

```python
cfg = TrainingConfig(
    MakeTimelaps=True,
    timelapse_output_folder=f"ResultsStudies/{Case_name}_timelapse_run",
)
```

When `timelapse_output_folder` is set, the trainer creates that folder and saves:

* `{case_name}_timelapse.avi`
* `best_result_frame.png`
* intermediate frame files under `timelapse_frames` while the video is being built

The timelapse layout contains:

* Top row: loading/boundary-condition image, front view, side view, and top view
* Bottom row: UV domain density distribution and 3D perspective material distribution
* Right panel: optimization losses and best-result summary

The video header includes key geometry and loading information next to the file name:

```text
Planar.stp (BBox: 10 x 10 x 0, F=0.01, SurfacePts=4332, FEM elements=5776)
```

The best-result frame also includes computation time as `Com. Time`.

## Loading And Boundary Visualization

`show_voxels_surface_and_bc()` visualizes:

* shell voxels
* surface samples
* fixed nodes
* loaded nodes
* load arrows

The returned image includes a compact top-left legend so it can be embedded cleanly in timelapse frames and best-result images.

## Practical Notes

* Seed updates must be **bounded and stable**
* FEM gradients can be **unstable and should be introduced gradually**
* Multi-face training requires proper normalization and careful gradient scaling
* Debugging should isolate the geometry path and FEM path
* Timelapse frames are useful for checking whether UV density, 3D material distribution, and boundary conditions remain consistent through training

## End Goal

A system that can:

* Automatically generate lattice structures
* Adapt to arbitrary surfaces
* Optimize both shape and mechanical behavior
* Remain fully differentiable and trainable

## Summary

This project combines **geometry**, **neural networks**, and **physics** into a single differentiable framework for learning-based structural design on surfaces.
