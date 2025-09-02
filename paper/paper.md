---
title: 'Optiland: Python-based, Open-Source Optical Design Software'
authors:
  - name: Kramer Harrison
    orcid: 0009-0000-5494-139X
    affiliation: 1
  - name: Manuel Fragata Mendes
    orcid: 0009-0009-2957-0799
    affiliation: 2
  - name: Corné Haasjes
    orcid: 0000-0003-0187-4116
    affiliation: "3, 4, 5"
  - name: Grégoire Hein
    orcid: 0009-0001-3029-7242
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
  - name: Friedrich Schiller University Jena
    index: 2
  - name: Department of Ophthalmology, Leiden University Medical Center, Leiden, The Netherlands
    index: 3
    ror: 05xvt9f17
  - name: Department of Radiology, Leiden University Medical Center, Leiden, The Netherlands
    index: 4
    ror: 05xvt9f17
  - name: Department of Radiation Oncology, Leiden University Medical Center, Leiden, The Netherlands
    index: 5
    ror: 05xvt9f17
date: 2025-09-02
bibliography: paper.bib
---

# Summary

**Optiland** is open-source optical design software written in Python. It offers a comprehensive platform for the design, analysis, and optimization of optical systems, catering to a wide audience from students and hobbyists to professional engineers and researchers. The software supports a variety of optical systems, including traditional refractive and reflective designs, as well as modern freeform and computational optics.

Core features include sequential ray tracing, a rich library of surface types (spherical, aspheric, freeform), optimization and tolerancing support, and a suite of analysis tools for evaluating optical performance (e.g., spot diagrams, wavefront analysis, modulation transfer function). A key feature of Optiland is its dual-backend architecture, which allows users to switch between a NumPy backend for fast CPU computations and a PyTorch backend for GPU acceleration and automatic differentiation. This enables the integration of Optiland with machine learning workflows and gradient-based optimization, as all calculations are differentiable. The software also includes a graphical user interface (GUI) for interactive design and analysis.

# Statement of Need

The field of optical design has long been dominated by commercial software tools such as OpticStudio and CODE V, which are powerful but expensive and proprietary. Licenses often cost tens of thousands of dollars, creating a significant barrier to entry for students, educators, and researchers.

Optiland addresses this need by providing the most complete open-source optical design package available. It enables a wide range of optical design, analysis, and optimization tasks that previously required costly commercial software. The differentiable PyTorch backend is particular relevant for computational optics and machine learning-driven design, where novel optimization and inverse-design approaches are increasingly important. For example, optical systems modeled in Optiland can be embedded into deep learning pipelines and trained end-to-end using backpropagation, enabling tasks such as lens design via learned generative models.

The PyTorch backend also provides significant performance gains through GPU acceleration. On typical modern hardware, GPU-accelerated ray tracing achieves speedups of 20-60x compared to CPU-bound NumPy computations, with greater gains possible on high-end or multi-GPU systems. This level of performance enables large-scale, gradient-based optimization and simulations for real-world research and development. By combining a flexible and fully differentiable architecture with strong performance and a rich feature set, Optiland aims to democratize access to advanced optical design tools.

Several open-source optical packages exist, such as Prysm [@Prysm], which provides advanced physical optics propagation and diffraction modeling, and RayOptics [@RayOptics], which offers Python-based ray-tracing and lens analysis. Optiland complements these efforts by combining ray tracing, optimization, tolerancing, and differentiable machine-learning integration into a single, comprehensive platform.

# Functionalities

Optiland supports a wide range of design, analysis, and optimization tasks, making it suitable for both classical optical engineering and modern computational applications. Its main capabilities include:

- **Design Tools**: Sequential ray tracing, lens system modeling (spherical, conic, aspheric, freeform surfaces), and flexible aperture/field/wavelength configurations.
- **Analysis Tools**: Spot diagrams, wavefront analysis, OPD maps, polarization ray tracing, PSF/MTF evaluation, and scattering models.
- **Optimization and Tolerancing**: Gradient-based and global optimization, Monte Carlo tolerancing, parametric sweeps, and specialized glass selection tools.
- **Differentiable Ray Tracing**: A fully differentiable backend via PyTorch enables gradient-based optimization and integration with machine learning frameworks.
- **Material Database**: Built-in refractive index library with support for user-defined materials.
- **Visualization**: 2D layout plots, 3D ray-trace visualization, and an interactive GUI.
- **Interoperability**: Import of Zemax OpticStudio files, JSON-based I/O, and a full Python API.
- **Performance**: GPU acceleration with PyTorch and CPU acceleration with Numba.

# Usage and Examples

The following examples demonstrate how to use Optiland, starting with a simple system definition, then optimization, and finally machine-learning integration with PyTorch.

## 1. Defining a simple optical system

```python
import numpy as np
from optiland import optic

# Create empty Optic instance
lens = optic.Optic()

# Define lens surfaces
lens.add_surface(index=0, thickness=np.inf)  # Object surface
lens.add_surface(index=1, thickness=7, radius=20.0, is_stop=True, material="N-SF11")
lens.add_surface(index=2, thickness=23.0, radius=-20.0)
lens.add_surface(index=3)  # Image surface

# Configure aperture, field, and wavelength
lens.set_aperture(aperture_type="EPD", value=20)
lens.set_field_type(field_type="angle")
lens.add_field(y=0)  # on-axis field
lens.add_wavelength(value=0.55, is_primary=True)

# Visualize in 3D
lens.draw3D()
```

![Singlet Lens in 3D](../docs/images/singlet.png)

## 2. Optimizing the lens

```python
from optiland import optimization

# Define optimization problem
problem = optimization.OptimizationProblem()
input_data = {
    "optic": lens,
    "surface_number": -1,  # image surface
    "Hx": 0, "Hy": 0,
    "num_rays": 5,
    "wavelength": 0.55,
    "distribution": "hexapolar",
}

problem.add_operand("rms_spot_size", target=0, weight=1, input_data=input_data)
problem.add_variable(lens, "radius", surface_number=1)
problem.add_variable(lens, "radius", surface_number=2)

# Optimize
optimizer = optimization.OptimizerGeneric(problem)
optimizer.optimize()
```

![Singlet Lens in 3D](../docs/images/singlet_optimized.png)

## 3. Switching to the PyTorch backend

Optiland’s API is identical across backends. Switching to PyTorch enables gradient tracking and GPU acceleration.

```python
import optiland.backend as be
be.set_backend("torch")      # Use the PyTorch backend
be.set_precision("float32")  # Set precision of calculations
be.grad_mode.enable()        # Enable gradient tracking
be.set_device("cuda")        # Use CUDA (GPU)
```

## 4. End-to-end optimization with PyTorch

The PyTorch backend allows integration with neural networks and gradient-based optimizers. Note that to enable this functionality, the lens should be built when the Pytorch backend is active.

```python
import torch
from optiland.ml import OpticalSystemModule

# Wrap the lens system and optimization problem in a PyTorch module
model = OpticalSystemModule(lens, problem)

# Define PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

losses = []
for step in range(250):
    optimizer.zero_grad()
    loss = model()      # merit function
    loss.backward()
    optimizer.step()
    model.apply_bounds()
    losses.append(loss.item())
```

This workflow enables research scenarios such as inverse design, generative optical modelling, or generic end-to-end optical design.

# Research Enabled by Optiland

Optiland is actively used by researchers in the **MReye group** at the Leiden University Medical Center. It serves as a configurable backend for all optical computations within the [Visisipy](https://github.com/MReye-LUMC/visisipy) [cite] project, a Python library for simulating visual optics.

# Figures

![Image of TMA, UV photolithography lens, and Folded Czerny-Turner Spectrometer](../docs/images/examples.png)

# Acknowledgements

The authors thank Jan-Willem Beenakker for initiating the collaboration that led to the integration of Optiland into Visisipy. We also thank contributors and community members for feedback and code contributions, in particular Seçkin Berkay Öztürk, Hemkumar Srinivas, Matteo Taccola, Corentin Nannini, and Kacper Rutkowski.

# References