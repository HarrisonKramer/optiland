.. _surface_overview:

Surface Overview
================

Surfaces are core elements of Optiland's optical system. Each surface represents an optical interface, defined by its geometry, material properties, coatings, and optional apertures. Surfaces are organized into a **Surface Group** to manage operations on multiple surfaces.

Surface Components
------------------

A surface consists of several components that define its optical properties:

- **Geometry**: The shape of the surface (e.g., planar, spherical, aspheric). Includes the surface's coordinate system.
- **Materials**: The refractive indices of the materials before and after the surface.
- **Coatings**: Thin-film coatings applied to the surface for modifying reflection and transmission properties.
- **Aperture (optional)**: A physical or virtual aperture defining the area where rays can interact with the surface.
- **Stop Surface Flag**: Indicates if the surface is the aperture stop of the system.
- **Physical Aperture** (optional): Defines the boundary for ray propagation.

Ray Interaction with Surfaces
-----------------------------

When a ray interacts with a surface:
1. **Intersection**: The ray's path is intersected with the surface's geometry.
2. **Refraction/Reflection**: The ray's direction is updated based on Snell's law and the surface's material/coating properties.
3. **Recording**: Data such as intersection points and modified ray attributes are stored for analysis.

The ray tracing process ensures accurate simulation of real-world optical behavior.

Surface Group
-------------

Surfaces are combined into a **Surface Group**, which manages a collection of surfaces and facilitates operations like ray tracing. The Surface Group:
- Tracks the ordered list of surfaces in the optical system.
- Propagates rays through the system, invoking surface-specific logic at each step.
- Records ray interactions for use in subsequent analyses.

.. tip::
   The Surface Group allows efficient iteration over multiple surfaces, simplifying complex ray tracing operations.

Surface Factory
---------------

To streamline surface creation, Optiland includes a **Surface Factory**. The factory:
- Generates the appropriate surface type based on user input.
- Adds the surface to the Surface Group at the specified position in the system.

Extensibility
-------------

The surface framework is designed for extensibility:
- Custom geometries, coatings, or aperture definitions can be added by subclassing existing components.
- The Surface Factory can be extended to handle new surface types.

For more detailed information on surface geometry and coatings, see their dedicated sections in this guide.
