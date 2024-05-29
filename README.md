[![License: GPL-3.0](https://img.shields.io/badge/License-GPL3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# **Optiland**

## Introduction
Optiland is a lens design and analysis program written in Python 3. It provides an intuitive and efficient interface for defining and visualizing lens systems, performing optimization of lens systems based on user-defined merit functions and variables, as well as analyzing optical systems using geometric and diffraction-based methods.

Get started immediately with [Optiland Tutorial #1](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_1a_Optiland_for_Beginners.ipynb) or see the extensive [Learning Guide](#learning-guide).

<img src="images/telephoto.png" alt="telephoto" width="800"/>


## Installation

The following steps can be used to install Optiland:

1. **Create and activate a virtual environment (optional, but recommended):**

    - Open a terminal or command prompt.
    - Navigate to your project directory.
    - Create a virtual environment by running:

        ```sh
        python -m venv venv
        ```

    - Activate the virtual environment:

        - On Windows:

            ```sh
            venv\Scripts\activate
            ```

        - On macOS and Linux:

            ```sh
            source venv/bin/activate
            ```

2. **Install the package using pip:**

    ```sh
    pip install git+https://github.com/HarrisonKramer/optiland.git
    ```


## Functionalities
- Lens entry
- 2D/3D visualization
- Paraxial analysis
- Real and paraxial ray tracing
- Real analysis functions (spot diagrams, ray aberration fans, OPD fans, distortion, MTF, etc.)
- Glass catalogue and index/abbe v-number determination (many thanks to refractiveindex.info)
- Design optimization
- Wavefront and Zernike analysis
- Coating analysis

The code itself is in constant flux and new functionalities are always being added.

## Learning Guide
This guide gives a step-by-step approach to learning how to use Optiland.

1. **Introduction to Optiland**
    - [Tutorial 1a - Optiland for Beginners](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_1a_Optiland_for_Beginners.ipynb)
         - Building your first lens system
         - Lens entry
         - Material selection
         - Aperture, field and wavelength selection
         - Drawing a lens in 2D and 3D
2. **Real Raytracing & Analysis**
    - [Tutorial 2a - Analyzing Ray Paths](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2a_Analyzing_Ray_Paths.ipynb)
        - Analyzing ray intersections & paths
        - Retrieving saved ray information after raytracing
    - [Tutorial 2b - Tilting & De-centering Components](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_2b_Tilting_&_Decentering_Components.ipynb)
        - Tracing rays through misaligned components
        - Adding a fold mirror to an optical system
3. **Aberrations**
    - [Tutorial 3a - Common aberration analyses/plots](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_3a_Common_Aberration_Analyses.ipynb)
        - Spot diagrams
        - Ray fans
        - Y-Ybar plots
        - Distortion / Grid distortion plots
        - Field curvature plots
    - [Tutorial 3b - 1st & 3rd Order Aberrations](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_3b_First_&_Third_Order_Aberrations.ipynb)
        - Calculation of seidel & 1st/3rd order aberrations
4. **PSF & MTF**
    - [Tutorial 4a - PSF & MTF Calculation](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_4a_PSF_&_MTF_Calculation.ipynb)
        - Geometric PSF/MTF
        - FFT-based PSF/MTF
5. **Optimization**
    - [Tutorial 5a - Simple Optimization](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_5a_Simple_Optimization.ipynb)
        - Operand and variable definition
        - Local optimization
        - How to improve a lens design
    - [Tutorial 5b - Advanced Optimization](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_5b_Advanced_Optimization.ipynb)
        - Global optimization
6. **Coatings & Polarization**
    - To be completed...
7. **Advanced Optical Design**
    - To be completed...
8. **Extending Optiland**
    - [Tutorial 8a - Custom Surface Types](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_8a_Custom_Surface_Types.ipynb)
        - Adding new surface types
    - [Tutorial 8b - Custom Coating Types](https://github.com/HarrisonKramer/optiland/blob/master/examples/Tutorial_8b_Custom_Coating_Types.ipynb)
        - Adding new coating types


## Acknowledgements & References
- https://github.com/Sterncat/opticspy
- https://github.com/jordens/rayopt
