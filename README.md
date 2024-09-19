![Tests](https://github.com/HarrisonKramer/optiland/actions/workflows/ci.yml/badge.svg?label=Tests)
[![codecov](https://codecov.io/github/HarrisonKramer/optiland/graph/badge.svg?token=KAOE152K5O)](https://codecov.io/github/HarrisonKramer/optiland)
![Stars](https://img.shields.io/github/stars/HarrisonKramer/optiland.svg)
![License](https://img.shields.io/badge/License-GPL3.0-blue.svg)

# **Optiland**

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#functionalities">Functionalities</a></li>
    <li><a href="#learning-guide">Learning Guide</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements-&-references">Acknowledgements & References</a></li>
  </ol>
</details>

## Introduction
Optiland is a lens design and analysis program written in Python 3. It provides an intuitive and efficient interface for defining and visualizing lens systems, performing optimization of lens systems based on user-defined merit functions and variables, as well as analyzing optical systems using geometric and diffraction-based methods. Leveraging computational libraries such as [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/), Optiland delivers exceptional performance and efficiency in handling complex optical computations.


Get started immediately with [Optiland Tutorial #1](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_1a_Optiland_for_Beginners.ipynb) or see the extensive [Learning Guide](https://github.com/HarrisonKramer/optiland?tab=readme-ov-file#learning-guide).


<figure style="text-align: center;">
  <img src="https://github.com/HarrisonKramer/optiland/raw/master/images/telephoto.png" alt="U.S. patent 2959100" style="width: 800px;">
</figure>

Python code to generate this 3D visualization:
```python
from optiland.samples.objectives import ReverseTelephoto
lens = ReverseTelephoto()
lens.draw3D()
```



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
    - [Tutorial 1a - Optiland for Beginners](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_1a_Optiland_for_Beginners.ipynb)
         - Lens entry
         - Material selection
         - Aperture, field and wavelength selection
         - Drawing a lens in 2D and 3D
    - [Tutorial 1b - Determining Lens Properties](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_1b_Lens_Properties.ipynb)
        - Focal length
        - Magnification
        - F-Number, Entrance/Exit pupil sizes & positions
        - Focal, Principal, and Nodal points, etc.
2. **Real Raytracing & Analysis**
    - [Tutorial 2a - Tracing & Analyzing Rays](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2a_Tracing_&_Analyzing_Rays.ipynb)
        - How to trace rays through a system
        - Analyzing ray paths & properties
    - [Tutorial 2b - Tilting & De-centering Components](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2b_Tilting_&_Decentering_Components.ipynb)
        - Tracing rays through misaligned components
    - [Tutorial 2c - Monte Carlo Raytracing Methods](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2c_Monte_Carlo_Raytracing.ipynb)
        - How variations in lens properties impact lens performance
    - [Tutorial 2d - Aspheric Components](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2d_Raytracing_Aspheres.ipynb)
        - Modeling even aspheres
3. **Aberrations**
    - [Tutorial 3a - Common aberration analyses/plots](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_3a_Common_Aberration_Analyses.ipynb)
        - Spot diagrams
        - Ray fans
        - Y-Ybar plots
        - Distortion / Grid distortion plots
        - Field curvature plots
    - [Tutorial 3b - 1st & 3rd-Order Aberrations](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_3b_First_&_Third_Order_Aberrations.ipynb)
        - Calculation of seidel, 1st & 3rd-order aberrations
    - [Tutorial 3c - Chromatic Aberrations](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_3c_Chromatic_Aberrations.ipynb)
        - Achromatic doublet to reduce chromatic aberrations
4. **Optical Path Difference (OPD), Point Spread Functions (PSF) & Modulation Transfer Function (MTF)**
    - [Tutorial 4a - Optical Path Difference](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_4a_Optical_Path_Difference_Calculation.ipynb)
        - OPD fans and plots
    - [Tutorial 4b - PSF & MTF Calculation](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_4b_PSF_&_MTF_Calculation.ipynb)
        - Geometric MTF
        - FFT-based PSF/MTF
    - [Tutorial 4c - Zernike Decomposition](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_4c_Zernike_Decomposition.ipynb)
        - Decomposing wavefront using Zernike polynomials
        - Coefficient types: Zernike standard, Zernike fringe, Zernike Noll
5. **Optimization**
    - [Tutorial 5a - Simple Optimization](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_5a_Simple_Optimization.ipynb)
        - Operand and variable definition
        - Local optimization
    - [Tutorial 5b - Advanced Optimization](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_5b_Advanced_Optimization.ipynb)
        - Global optimization
    - [Tutorial 5c - Optimization Case Study](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_5c_Optimization_Case_Study.ipynb)
        - Complete process of designing a Cooke triplet
6. **Coatings & Polarization**
    - [Tutorial 6a - Introduction to Coatings](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_6a_Introduction_to_Coatings.ipynb)
        - Simple coatings in Optiland
        - Impact of coatings on system performance
    - [Tutorial 6b - Introduction to Polarization](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_6b_Introduction_to_Polarization.ipynb)
        - Basics of polarization in Optiland
        - Analyzing polarization performance
    - Tutorial 6c - Advanced Polarization - Update in progress (target completion: Nov. 2024)
        - Waveplates, polarizers, and the Jones matrix
        - Jones pupils
7. **Advanced Optical Design**
    - [Tutorial 7a - Lithographic Projection System](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_7a_Lithographic_Projection_System.ipynb)
        - Optimizing and Analyzing a Complex Lithography System
    - [Tutorial 7b - Surface Roughness & Scattering](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_7b_Surface_Roughness_&_Scattering.ipynb)
        - Lambertian and Gaussian scatter models
    - [Tutorial 7c - Freeform Surfaces](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_7c_Freeform_Surfaces.ipynb)
        - Designing non-standard optical systems with freeform surfaces
8. **Tolerancing**
    - [Tutorial 8a - Introduction to Tolerancing](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_8a_Tolerancing_Sensitivity_Analysis.ipynb)
        - Sensitivity studies
    - [Tutorial 8b - Advanced Tolerancing](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_8b_Monte_Carlo_Analysis.ipynb)
        - Monte Carlo-based Tolerancing
9. **Lens Catalogue Integration**
    - [Tutorial 9a - Edmund Optics Catalogue](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_9a_Edmund_Optics_Catalogue.ipynb)
        - Reading Zemax files
        - Reading and analyzing an aspheric lens
    - [Tutorial 9b - Thorlabs Catalogue](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_9b_Thorlabs_Catalogue.ipynb)
        - Reading and analyzing an achromatic doublet pair lens
10. **Extending Optiland**
    - Tutorial 10a - Custom Surface Types - Update in progress (target completion: Nov. 2024)
        - Adding new surface types
    - Tutorial 10b - Custom Coating Types - Update in progress (target completion: Nov. 2024)
        - Adding new coating types
    - Tutorial 10c - Custom Optimization Algorithms - Update in progress (target completion: Nov. 2024)
        - Adding new optimization approaches
11. **Machine Learning in Optical Design**
    - Tutorial 11a - Reinforcement Learning for Lens Design - Update in progress (target completion: Nov. 2024)


## License
Distributed under the GPL-3.0 License. See [LICENSE](https://github.com/HarrisonKramer/optiland/blob/master/LICENSE) for more information.


## Contact
Kramer Harrison - kdanielharrison@gmail.com

## Acknowledgements & References
- https://www.lens-designs.com/
- https://github.com/Sterncat/opticspy
- https://github.com/jordens/rayopt
