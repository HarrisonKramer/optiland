Ray Overview
================

Ray Tracing Framework
=====================
The ray tracing framework supports a variety of ray types and provides tools for simulating ray propagation through the optical system:

- **Ray Types**:
  - *Real Rays*: Trace actual paths based on Snell's law and system geometry.
  - *Paraxial Rays*: Simplified tracing for small-angle approximations.
  - *Polarized Rays*: Include polarization effects for advanced simulations.
- **Ray Generator**: Handles ray initialization for different field points and wavelengths.

The modular design allows for customized ray tracing configurations, enabling high flexibility in simulations.