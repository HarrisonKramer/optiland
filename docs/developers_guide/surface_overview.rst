Surface Overview
================

Surface Definition
------------------
Surfaces in Optiland are fundamental to the definition of optical systems. Each surface is responsible for representing a discrete optical interface. Key attributes of a surface include:

- **Geometry**: Defines the surface shape and is associated with a coordinate system.
- **Coatings**: Specifies thin film properties for the surface.
- **Materials**: The refractive index of the material before and after the surface.
- **Stop Flag**: Indicates whether the surface is the aperture stop of the system.
- **Physical Aperture** (optional): Defines the boundary for ray propagation.

Each surface belongs to the `Surface Group` within the `Optic` class, which maintains the sequence of surfaces in the optical system.