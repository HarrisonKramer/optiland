.. _functionalities:

Optiland provides a comprehensive suite of tools for the design, analysis, and optimization of optical systems. Below is an overview of its key functionalities:

Design Tools
------------
- **Sequential Ray Tracing**: Handle both traditional lens or mirror designs and more complex asymmetric or freeform systems.
- **Lens System Modeling**: Support for spherical, conical, aspheric, and freeform surfaces.
- **Aperture, Field and Wavelength Settings**: Configure optical systems for diverse apertures, fields, and wavelengths.

Analysis Tools
--------------
- **Real Ray and Paraxial Ray Tracing**: Perform analyses for both idealized and real optical systems.
- **Polarization Ray Tracing**: Model polarization-dependent effects with full vector calculations.
- **Full Analysis Suite**: Generate spot diagrams, ray aberration fans, optical path difference (OPD) fans, distortion curves, and more.
- **Wavefront Analysis**: Includes Zernike decomposition, wavefront error evaluation, and wavefront maps.
- **Point Spread Function (PSF) and Modulation Transfer Function (MTF)**: Evaluate imaging system performance.
- **BSDF and Scattering Analysis**: Evaluate surface scatter using Bidirectional Scattering Distribution Functions (BSDF).

Optimization and Tolerancing Tools
----------------------------------
- **Local and Global Optimization**: Includes gradient-based methods, evolutionary algorithms, and global search routines for design optimization.
- **Operand-Based Framework**: Support for user-defined performance metrics and merit functions.
- **Tolerancing Analysis**: Perform sensitivity analysis, Monte Carlo simulations, and manufacturability studies.
- **Optimization Extension**: Extend the optimization framework with custom operands, variables, or optimization algorithms.

Material Database
-----------------
- **Extensive Material Library**: Integrated access to refractive index data from refractiveindex.info.
- **Custom Material Models**: Support for user-defined material models and dispersion formulas.

Visualization
-------------
- **2D and 3D Visualizations**: Dynamic visualization of optical systems in 2D and 3D via matplotlib and VTK.
- **Interactive Tools**: Fully interactive visualizations for analysis and debugging.

Compatibility and Interoperability
----------------------------------
- **Zemax File Import**: Load and adapt optical systems designed in Zemax.
- **File Import/Export**: Save and load optical systems in JSON format for easy sharing and interoperability.
- **Python API**: Automate tasks and extend functionality with Python scripting.

Performance
-----------
- **Ray Tracing Speed**: On a typical modern machine, Optiland achieves speeds of approximately **5 to 10 million ray surfaces per second**, depending on system complexity and ray properties such as polarization.
- **Use of NumPy and Numba**: Optiland leverages NumPy for array operations and Numba for just-in-time compilation to achieve high performance.

.. note::
   If there are any functionalities you would like to see in Optiland, please let us know by opening an issue on our GitHub repository.
