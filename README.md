![Tests](https://github.com/HarrisonKramer/optiland/actions/workflows/ci.yml/badge.svg?label=Tests)
[![Documentation Status](https://readthedocs.org/projects/optiland/badge/?version=latest)](https://optiland.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/github/HarrisonKramer/optiland/graph/badge.svg?token=KAOE152K5O)](https://codecov.io/github/HarrisonKramer/optiland)
[![Maintainability](https://api.codeclimate.com/v1/badges/2fa0f839a0f3dbc4d5b1/maintainability)](https://codeclimate.com/github/HarrisonKramer/optiland/maintainability)
![Stars](https://img.shields.io/github/stars/HarrisonKramer/optiland.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14588961.svg)](https://doi.org/10.5281/zenodo.14588961)


<div align="center">
  <a href="https://optiland.readthedocs.io/">
    <img src="https://github.com/HarrisonKramer/optiland/raw/master/docs/images/optiland.svg" alt="Optiland">
  </a>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#documentation">Documentation</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#functionalities">Functionalities</a></li>
    <li><a href="#learning-guide">Learning Guide</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact-and-support">Contact and Support</a></li>
  </ol>
</details>

## Introduction
**Optiland** is a Python-based, open-source optical design and analysis framework. With a simple and intuitive Python interface, Optiland enables the design, optimization, and analysis of complex optical systems, from paraxial and real raytracing to polarization, coatings, and wavefront analyses. It supports 2D/3D visualization, comprehensive tolerancing, local and global optimization, and freeform optics, among other features. Built on the speed of [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/), Optiland delivers computational efficiency and flexibility across a wide range of optical tasks.

Get started immediately with [Optiland Tutorial #1](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1a_Optiland_for_Beginners.ipynb), see the extensive [Learning Guide](https://github.com/HarrisonKramer/optiland?tab=readme-ov-file#learning-guide), or read the full documentation at [Read the Docs](https://optiland.readthedocs.io/).


<figure style="text-align: center;">
  <img src="https://github.com/HarrisonKramer/optiland/raw/master/docs/images/telephoto.png" alt="U.S. patent 2959100" style="width: 800px;">
</figure>

Python code to generate this 3D visualization:
```python
from optiland.samples.objectives import ReverseTelephoto
lens = ReverseTelephoto()
lens.draw3D()
```

## Documentation
The full documentation for Optiland is hosted on [Read the Docs](https://optiland.readthedocs.io/).

Explore the [Example Gallery](https://optiland.readthedocs.io/en/latest/gallery/introduction.html) for a wide range of lens designs and analyses created with Optiland.

See the [Developer's Guide](https://optiland.readthedocs.io/en/latest/developers_guide/introduction.html) for an extensive overview of the architecture and design of Optiland and the [API Reference](https://optiland.readthedocs.io/en/latest/api/api_introduction.html) for detailed documentation of all public classes, methods, and functions.

## Installation

You can install the package using pip. To do so, follow these steps:

1. Open a terminal or command prompt.
2. Run the following command to install the package:

    ```sh
    pip install optiland
    ```


## Functionalities
- Lens entry
- 2D/3D visualization
- Paraxial and aberration analyses
- Real and paraxial ray tracing, including aspherics and freeforms
- Polarization ray tracing
- Real analysis functions (spot diagrams, ray aberration fans, OPD fans, distortion, PSF, MTF, etc.)
- Glass and material catalogue (based on refractiveindex.info)
- Design optimization (local and global)
- Wavefront and Zernike analysis
- Tolerancing, including sensitivity analysis and Monte Carlo methods
- Coating and surface scatter (BSDF) analysis
- Zemax file import

The code itself is in constant flux and new functionalities are always being added.

## Learning Guide
This guide gives a step-by-step approach to learning how to use Optiland.

1. **Introduction to Optiland**
    - [Tutorial 1a - Optiland for Beginners](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1a_Optiland_for_Beginners.ipynb)
         - Lens entry
         - Material selection
         - Aperture, field and wavelength selection
         - Drawing a lens in 2D and 3D
    - [Tutorial 1b - Determining Lens Properties](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1b_Lens_Properties.ipynb)
        - Focal length
        - Magnification
        - F-Number, Entrance/Exit pupil sizes & positions
        - Focal, Principal, and Nodal points, etc.
    - [Tutorial 1c - Saving and Loading Lenses](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1c_Save_and_Load_Files.ipynb)
        - Saving and loading lens files in a json format
    - [Tutorial 1d - Using the Material Database](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1d_Material_Database.ipynb)
        - Defining materials for glass, chemicals, organics, gases, or using ideal or parameterized materials.
    - [Tutorial 1e - Non-rotationally Symmetric Systems](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_1e_Non_Rotationally_Symmetric_Systems.ipynb)
        - Coordinate systems in Optiland and how to design non-rotationally symmetric systems
2. **Real Raytracing & Analysis**
    - [Tutorial 2a - Tracing & Analyzing Rays](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_2a_Tracing_&_Analyzing_Rays.ipynb)
        - How to trace rays through a system
        - Analyzing ray paths & properties
    - [Tutorial 2b - Tilting & De-centering Components](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_2b_Tilting_&_Decentering_Components.ipynb)
        - Tracing rays through misaligned components
    - [Tutorial 2c - Monte Carlo Raytracing Methods](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_2c_Monte_Carlo_Raytracing.ipynb)
        - How variations in lens properties impact lens performance
    - [Tutorial 2d - Aspheric Components](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_2d_Raytracing_Aspheres.ipynb)
        - Modeling even aspheres
3. **Aberrations**
    - [Tutorial 3a - Common aberration analyses/plots](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_3a_Common_Aberration_Analyses.ipynb)
        - Spot diagrams
        - Ray fans
        - Y-Ybar plots
        - Distortion / Grid distortion plots
        - Field curvature plots
        - RMS spot size & wavefront error vs. field
        - Pupil aberration
    - [Tutorial 3b - 1st & 3rd-Order Aberrations](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_3b_First_&_Third_Order_Aberrations.ipynb)
        - Calculation of seidel, 1st & 3rd-order aberrations
    - [Tutorial 3c - Chromatic Aberrations](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_3c_Chromatic_Aberrations.ipynb)
        - Achromatic doublet to reduce chromatic aberrations
4. **Optical Path Difference (OPD), Point Spread Functions (PSF) & Modulation Transfer Function (MTF)**
    - [Tutorial 4a - Optical Path Difference](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_4a_Optical_Path_Difference_Calculation.ipynb)
        - OPD fans and plots
    - [Tutorial 4b - PSF & MTF Calculation](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_4b_PSF_&_MTF_Calculation.ipynb)
        - Geometric MTF
        - FFT-based PSF/MTF
    - [Tutorial 4c - Zernike Decomposition](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_4c_Zernike_Decomposition.ipynb)
        - Decomposing wavefront using Zernike polynomials
        - Coefficient types: Zernike standard, Zernike fringe, Zernike Noll
5. **Optimization**
    - [Tutorial 5a - Simple Optimization](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_5a_Simple_Optimization.ipynb)
        - Operand and variable definition
        - Local optimization
    - [Tutorial 5b - Advanced Optimization](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_5b_Advanced_Optimization.ipynb)
        - Global optimization
    - [Tutorial 5c - Optimization Case Study](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_5c_Optimization_Case_Study.ipynb)
        - Complete process of designing a Cooke triplet
    - [Tutorial 5d - User-defined Optimization Metrics](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_5d_User_Defined_Optimization.ipynb)
        - Customized optimization
6. **Coatings & Polarization**
    - [Tutorial 6a - Introduction to Coatings](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_6a_Introduction_to_Coatings.ipynb)
        - Simple coatings in Optiland
        - Impact of coatings on system performance
    - [Tutorial 6b - Introduction to Polarization](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_6b_Introduction_to_Polarization.ipynb)
        - Basics of polarization in Optiland
        - Analyzing polarization performance
7. **Advanced Optical Design**
    - [Tutorial 7a - Lithographic Projection System](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_7a_Lithographic_Projection_System.ipynb)
        - Optimizing and Analyzing a Complex Lithography System
    - [Tutorial 7b - Surface Roughness & Scattering](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_7b_Surface_Roughness_&_Scattering.ipynb)
        - Lambertian and Gaussian scatter models
    - [Tutorial 7c - Freeform Surfaces](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_7c_Freeform_Surfaces.ipynb)
        - Designing non-standard optical systems with freeform surfaces
    - [Tutorial 7d - Three Mirror Anastigmat](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_7d_Three_Mirror_Anastigmat.ipynb)
        - Off-axis reflective telescope with freeform surfaces
8. **Tolerancing**
    - [Tutorial 8a - Introduction to Tolerancing](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_8a_Tolerancing_Sensitivity_Analysis.ipynb)
        - Sensitivity studies
    - [Tutorial 8b - Advanced Tolerancing](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_8b_Monte_Carlo_Analysis.ipynb)
        - Monte Carlo-based Tolerancing
9. **Lens Catalogue Integration**
    - [Tutorial 9a - Edmund Optics Catalogue](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_9a_Edmund_Optics_Catalogue.ipynb)
        - Reading Zemax files
        - Reading and analyzing an aspheric lens
    - [Tutorial 9b - Thorlabs Catalogue](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_9b_Thorlabs_Catalogue.ipynb)
        - Reading and analyzing an achromatic doublet pair lens
10. **Extending Optiland**
    - [Tutorial 10a - Custom Surface Types](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_10a_Custom_Surface_Types.ipynb)
        - Adding new surface types
    - [Tutorial 10b - Custom Coating Types](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_10b_Custom_Coating_Types.ipynb)
        - Adding new coating types
    - [Tutorial 10c - Custom Optimization Algorithms](https://github.com/HarrisonKramer/optiland/blob/master/docs/examples/Tutorial_10c_Custom_Optimization_Algorithm.ipynb)
        - Creating a "random walk optimizer" to optimize an aspheric singlet
11. **Machine Learning in Optical Design** - note that these notebooks are hosted in the [LensAI repository](https://github.com/HarrisonKramer/LensAI)
    - [Tutorial 11a - Random Forest Regressor to Predict Optimal Lens Properties](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_1/Singlet_RF_Model_RMS_Spot_Size.ipynb)
        - Demonstrates how to build and train a random forest regressor to predict the radius of curvature of a plano-convex lens in order to minimize the RMS spot size.
    - [Tutorial 11b - Ray Path Failure Classification Model](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_2/Ray_Path_Failure_Classification_Model.ipynb)
        - Uses logistic regression to predict ray path failures in a Cooke triplet design.
    - [Tutorial 11c - Surrogate Ray Tracing Model Using Neural Networks](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_3/Double_Gauss_Surrogate_Model.ipynb)
        - Builds a neural network surrogate ray tracing model to increase effective "ray tracing" speed by 10,000x.
    - [Tutorial 11d - Super-Resolution Generative Adversarial Network to Enhance Wavefront Map Data](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_5/SR_GAN_for_wavefront_data.ipynb)
        - Utilizes a super-resolution GAN (SRGAN) to upscale low-resolution wavefront data into high-resolution data.
    - [Tutorial 11e - Optimization of Aspheric Lenses via Reinforcement Learning](https://github.com/HarrisonKramer/LensAI/blob/main/notebooks/Example_4/RL_aspheric_singlet.ipynb)
        - Reinforcement learning is applied to the optimization of aspheric singlet lenses to generate new lens designs.


## Roadmap

Optiland is continually evolving to provide new functionalities for optical design and analysis. Below are some of the planned features and enhancements we aim to implement in future versions:

- [ ] **GUI (PySide6-based)**
- [ ] **Multiple Configurations (Zoom Lenses)**
- [ ] **Thin Film Design and Optimization** 
- [ ] **Diffractive Optical Elements**
- [ ] **Differentiable Ray Tracer via PyTorch**
- [ ] **Configurable Backends: NumPy, PyTorch, CuPy**
- [ ] **Jones Pupils**
- [ ] **Apodization Support** 
- [ ] **Additional Freeforms (Superconic, etc.)**
- [ ] **Image Simulation**
- [ ] **Huygens PSF & MTF**
- [ ] **Interferogram Analysis**
- [ ] **Additional Tutorials/Examples**
- [ ] **Non-sequential ray tracing**
- [ ] **Glass Expert**
- [ ] **Insert your idea here...**


### Community Contributions
We welcome suggestions for additional features! If there's something you'd like to see in Optiland, feel free to open an issue or discussion.


## License
Distributed under the MIT License. See [LICENSE](https://github.com/HarrisonKramer/optiland/blob/master/LICENSE) for more information.


## Contact and Support
If you have questions, find a bug, have suggestions for new features, or need help, please [open an issue](https://github.com/HarrisonKramer/optiland/issues) in the GitHub repository. This ensures that your concern is visible to others, can be discussed collaboratively, and helps build a public archive of solutions for similar inquiries in the future.

While I prefer issues as the primary means of communication, you may also contact me via email if necessary.

Kramer Harrison - kdanielharrison@gmail.com
