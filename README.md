[![License: GPL-3.0](https://img.shields.io/badge/License-GPL3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# Optiland

## Introduction
Optiland is a lens design and analysis program written in Python 3. It provides an intuitive and efficient interface for defining and visualizing lens systems, performing optimization of lens systems based on user-defined merit functions and variables, as well as analyzing optical systems using geometric and diffraction-based methods.

<img src="images/telephoto.png" alt="telephoto" width="800"/>

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

## Getting started
The best way to get started with this software is to check out the several Jupyter notebooks located in this repository. Cooke_Triplet.ipynb demonstrates many of the functionalities.

## Acknowledgements & References
- https://github.com/Sterncat/opticspy
- https://github.com/jordens/rayopt
