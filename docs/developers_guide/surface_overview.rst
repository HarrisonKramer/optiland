.. _surface_overview:

Surface Overview
================

Surfaces are core elements of Optiland's optical system. Each surface represents an optical interface, defined by its geometry,
material properties, coatings, and optional apertures. Surfaces are organized into a **Surface Group** to manage operations on multiple surfaces.

Surface Components
------------------

A surface consists of several components that define its optical properties:

- **Geometry**: The shape of the surface (e.g., planar, spherical, aspheric, freeform). This includes the surface's coordinate system.
- **Materials**: The material type before and after the surface, which determines the refractive index and extinction coefficient.
- **Coatings**: Coatings (e.g., thin films) applied to the surface for modifying reflection, transmission, and/or polarization properties.
- **Stop Surface Flag**: Indicates if the surface is the aperture stop of the system.
- **Reflective Flag**: Indicates if the surface is reflective (e.g., mirror) or refractive (e.g., lens).
- **Physical Aperture (optional)**: A physical or virtual aperture defining the area where rays can interact with the surface.
- **BSDF** (optional): Bidirectional Scattering Distribution Function for modeling scattering behavior.

Paraxial Surfaces
-----------------

A special surface type is the **Paraxial Surface**, which is a simplified ideal lens defined only by its focal length (assumed in air). This surface can be used for first order layouts
at the beginning of the optical design process. See the :ref:`gallery_basic_lenses` for an example of using paraxial surfaces. Under the hood, paraxial surfaces are defined as planar
surfaces and use simplified ray tracing logic, which assumes paraxial behavior.

Ray Interaction with Surfaces
-----------------------------

When a ray interacts with a surface, the following steps are typically performed:

1. **Intersection**: The ray's path is intersected with the surface's geometry.
2. **Aperture Check**: The ray's intersection point is checked against the surface's aperture to determine if the ray is blocked.
3. **Refraction/Reflection**: The ray's direction is updated based on Snell's law or the law of reflection, and the ray properties may be affected by the surface's material/coating properties.
4. **Scattering**: If the surface has a BSDF, the ray may be scattered based on the scattering distribution function.
5. **Recording**: Data such as intersection points and modified ray attributes are stored for later analysis.

Surface Group
-------------

Surfaces are combined into a **Surface Group**, which manages a collection of surfaces and facilitates operations like ray tracing. The Surface Group:

- Tracks the ordered list of surfaces in the optical system.
- Propagates rays through the system, invoking surface-specific logic at each step.
- Exposes methods for adding, removing, and modifying surfaces in the system.

.. tip::
   The Surface Group allows efficient iteration over multiple surfaces, simplifying complex ray tracing operations.

Surface Factory
---------------

To streamline surface creation, Optiland includes a **Surface Factory**. The factory:

- Generates the appropriate surface type based on user input.
- Configures the surface with the specified geometry, material, coatings, and other properties.
- Adds the surface to the Surface Group at the specified position in the system, based on the surface index.

Extensibility
-------------

The surface framework is designed for extensibility:

- Custom geometries, coatings, or aperture definitions can be added by subclassing existing components. These may be added to any surface instance.
- The Surface Factory can be extended to handle new surface types.

For more detailed information on surface geometry and coatings, see their dedicated sections in this guide.
