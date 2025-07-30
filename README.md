![Tests](https://github.com/HarrisonKramer/optiland/actions/workflows/ci.yml/badge.svg?label=Tests)
[![Documentation Status](https://readthedocs.org/projects/optiland/badge/?version=latest)](https://optiland.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/HarrisonKramer/optiland/graph/badge.svg?token=KAOE152K5O)](https://codecov.io/github/HarrisonKramer/optiland)
[![Maintainability](https://qlty.sh/gh/HarrisonKramer/projects/optiland/maintainability.svg)](https://qlty.sh/gh/HarrisonKramer/projects/optiland)
![Stars](https://img.shields.io/github/stars/HarrisonKramer/optiland.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14588961.svg)](https://doi.org/10.5281/zenodo.14588961)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%7C%203.12%20%7C%203.13%20-blue)](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%7C%203.12%20%7C%203.13%20-blue)


<div align="center">
  <a href="https://optiland.readthedocs.io/">
    <img src="https://github.com/HarrisonKramer/optiland/raw/master/docs/images/optiland.svg" alt="Optiland">
  </a>
</div>

<div align="center">
    <img src="https://github.com/HarrisonKramer/optiland/raw/master/docs/images/gui.png" alt="Optiland GUI" style="max-width: 100%; height: auto;">
  </a>
</div>

<p align="center"><em>The Optiland GUI showing a reverse telephoto system.</em></p>


## Contents

1. [Introduction](#introduction)
2. [Documentation](#documentation)
3. [Installation](#installation)
4. [Core capabilities](#core-capabilities)
5. [Learning Guide](#learning-guide)
6. [Roadmap](#roadmap)
6. [Under development](#under-development)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact and Support](#contact-and-support)


---

## Introduction

**Optiland** is an open-source optical design platform built in Python, tailored for both classical lens systems and modern computational optics. It provides a robust and extensible interface for constructing, optimizing and analyzing optical systems, from standard refractive or reflective layouts to advanced freeform assemblies.

Built for professional engineering workflows, Optiland includes full support for tolerancing, high-performance optimization routines, and intelligent material selection through its integrated GlassExpert module. Also, traditional ray-based analysis are complemented by differentiable modeling support through PyTorch.

Whether you're developing prototypes in research or refining production systems, Optiland delivers the flexibility and precision needed to model, simulate, and optimize real-world optical instruments:

- ⚙️ Build reractive and reflective systems using a clean, object-oriented API  
- 🔍 Trace rays through multi-surface optical assemblies, including aspherics and freeforms
- 📊 Analyze paraxial properties, wavefront errors, PSFs/MTFs, and scatter behavior 
- 🧠 Optimize via traditional merit functions or autograd-enabled differentiable backends  
- 🎨 Visualize interactively in 2D (Matplotlib) and 3D (VTK).

Under the hood, Optiland uses NumPy for fast CPU calculations and PyTorch for GPU acceleration and automatic differentiation. Switch between engines depending on your use case with the same interface.

**Quickstart**  
1. [Quickstart Tutorial](https://optiland.readthedocs.io/en/latest/examples/Tutorial_1a_Optiland_for_Beginners.html) – build your first lens in 5 minutes  
2. [Full Learning Guide](https://optiland.readthedocs.io/en/latest/learning_guide.html) – in-depth guide to mastering Optiland 
3. [Example Gallery](https://optiland.readthedocs.io/en/latest/gallery/introduction.html) – visual showcase of designs and core features
4. [Cheat Sheet](https://optiland.readthedocs.io/en/latest/cheat_sheet.html) - an up-to-date cheat sheet to get you started ASAP with your first optical system

---

## Documentation

Optiland's full documentation is available on [Read the Docs](https://optiland.readthedocs.io/).



## Installation

- **Core only**

    ```bash
    pip install optiland
    ```

- **Core + GUI**

    ```bash
    pip install optiland[gui]
    ```

- **With CPU‑only PyTorch**

    ```bash
    pip install optiland[torch]
    ```

- **GPU‑enabled PyTorch**

    After installing Optiland, install a CUDA build of PyTorch manually:

    ```bash
    pip install optiland
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```


For more details, see the [installation guide](https://optiland.readthedocs.io/en/latest/installation.html) in the docs.

## Core Capabilities

| Feature       | Capabilities |
|------------------------|--------------|
| **🛠️ Design & Modeling** | Configure fields, wavelengths, apertures. Build systems using spherical, aspheric, conic, and freeform surfaces.  |
| **🧮 Differentiable Core** | Switch between NumPy (CPU) and PyTorch (GPU/autograd) seamlessly for hybrid physics-ML workflows. |
| **🔬 Ray Tracing** | Trace paraxial and real rays through sequential systems with support for polarization, birefringence, and coatings. |
| **📊 Optical Analysis** | Generate spot diagrams, wavefront error maps, ray fans, PSF/MTF plots, Zernike decompositions, distortion plots, etc. |
| **🧠 Optimization** | Local & global optimizers, autograd support, operand-based merit functions, and GlassExpert for categorical variable search. |
| **📈 Tolerancing** | Monte Carlo and parametric sensitivity analysis to evaluate robustness and manufacturability. |
| **📚 Material Library** | Integrated access to refractiveindex.info. Support for custom dispersion models and material creation. |
| **🖼️ Visualization** | 2D plots via matplotlib, 3D interactive scenes with VTK, and debugging tools to inspect ray behavior. |
| **🧩 Interoperability** | Import Zemax files, save/load systems in JSON, use full Python API for scripting and automation. |
| **🚀 Performance** | GPU-accelerated ray tracing (150M+ ray-surfaces/s), Numba-optimized NumPy backend, JIT-compiled computations. |
| **🤖 ML Integration** | Compatible with PyTorch pipelines for deep learning, differentiable modeling, and end-to-end training. |


For a full breakdown of Optiland’s functionalities, see the [complete feature list](https://optiland.readthedocs.io/en/latest/functionalities.html).

> [!NOTE]
> The code itself is in constant flux and new functionalities are always being added.



## Roadmap

Optiland is continually evolving to provide new functionalities for optical design and analysis. Below are some of the planned features and enhancements we aim to implement in future versions:

- [x] **GUI (PySide6-based)** - *Initial version available, ongoing enhancements.*
- [ ] **Multi-Path Sequential Ray Tracing**
- [ ] **Multiple Configurations (Zoom Lenses)**
- [ ] **Thin Film Design and Optimization** 
- [ ] **Diffractive Optical Elements**
- [ ] **Additional Backends: JAX, CuPy**
- [ ] **Jones Pupils**
- [ ] **Additional Freeforms (Superconic, etc.)**
- [ ] **Image Simulation**
- [ ] **Huygens MTF**
- [ ] **Interferogram Analysis**
- [ ] **Additional Tutorials/Examples**
- [ ] **Non-sequential ray tracing**
- [ ] **Insert your idea here...**


## Under Development

Welcome, contributors! This section outlines the major features and tasks that are currently in progress. To avoid duplicated effort, **please check this table and the [GitHub Issues](https://github.com/HarrisonKramer/optiland/issues)** before starting work. If you’d like to work on something, **comment on the issue to let others know.** You can find more about how to coordinate in our [contributing guide](./CONTRIBUTING.md).

| Feature / Topic | Contributor(s) | Status | Discussion / Issue |
| ------------------------------------------------ | -------------------------------------------------- | -------------- | ------------------------------------------------ |
| **Core** |                                                    |                |                                                  |
| Forbes Surface Type | [@manuelFragata](https://github.com/manuelFragata) | 🚧 In Progress | - |
| (extended) Sources | [@manuelFragata](https://github.com/manuelFragata) | 🚧 In Progress | [#224](https://github.com/HarrisonKramer/optiland/issues/224) |
| Multi Sequence Tracing | [@HarrisonKramer](https://github.com/HarrisonKramer) | 🔍 Under Review | [#89](https://github.com/HarrisonKramer/optiland/issues/89)   |
| Image Simulation Analysis | [@HarrisonKramer](https://github.com/HarrisonKramer) | 🚧 In Progress | [#153](https://github.com/HarrisonKramer/optiland/issues/153) |
| Diffraction Gratings and DOEs| [@Hhsoj](https://github.com/Hhsoj) [@mattemilio](https://github.com/mattemilio) | 🚧 In Progress | [#161](https://github.com/HarrisonKramer/optiland/issues/161) [#188](https://github.com/HarrisonKramer/optiland/discussions/188) [#225](https://github.com/HarrisonKramer/optiland/issues/225) |
| Paraxial to Thick Lens | [@HarrisonKramer](https://github.com/HarrisonKramer) | 🚧 In Progress | [#221](https://github.com/HarrisonKramer/optiland/issues/221) |
| **Analysis & Visualization** |                                                    |                |                                                  |
| Huygens MTF | [@HarrisonKramer](https://github.com/HarrisonKramer)| 🚧 In Progress | [#222](https://github.com/HarrisonKramer/optiland/issues/222) |
| Sag Surface Analysis | [@manuelFragata](https://github.com/manuelFragata)| ✅ Done | [#183](https://github.com/HarrisonKramer/optiland/issues/183) |
| **GUI** |                                                    |                |                                                  |
| GUI First Iteration | [@manuelFragata](https://github.com/manuelFragata)| ✅ Done | - |
| GUI - Console/Terminal | [@manuelFragata](https://github.com/manuelFragata)| ✅ Done | - |


**Status**
* ✨ **Help Wanted**: We are actively looking for contributors for this task!
* 🚧 **In Progress**: Actively being worked on.
* 🔍 **Under Review**: A pull request has been submitted and is being reviewed.
* 🛑 **Blocked**: Progress is blocked by another issue.
* ✅ **Done**: Completed and merged. (You can remove these after a while).


## Contributing

We welcome contributions of all kinds — features, bugfixes, docs, and discussions! 🎉

To get started, please check out the [contributing guide](./CONTRIBUTING.md) for best practices and coordination tips.


## License
Distributed under the MIT License. See [LICENSE](https://github.com/HarrisonKramer/optiland/blob/master/LICENSE) for more information.


## Contact and Support
If you have questions, find a bug, have suggestions for new features, or need help, please [open an issue](https://github.com/HarrisonKramer/optiland/issues) in the GitHub repository. This ensures that your concern is visible to others, can be discussed collaboratively, and helps build a public archive of solutions for similar inquiries in the future.

While I prefer issues as the primary means of communication, you may also contact me via email if necessary.

Kramer Harrison - kdanielharrison@gmail.com
