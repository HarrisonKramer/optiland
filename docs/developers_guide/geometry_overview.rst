Geometry Overview
=================

Geometries define the shapes of optical surfaces and play a key role in ray tracing by handling two critical operations:

1. **Ray-Surface Intersection**: Determining the intersection point (or propagation distance) between a ray and the surface.
2. **Surface Normal Calculation**: Computing the surface normal at the intersection point, which is essential for applying Snell's law or the law of reflection.

Key Components
--------------

A geometry is defined by:

- **Coordinate System**: Each geometry relies on a `CoordinateSystem` object, which specifies:

  - Position: `x`, `y`, `z` (surface origin in the global coordinate space)
  - Rotation: `rx`, `ry`, `rz` (Euler angles or equivalent rotation representation)
  - Reference Coordinate System: Enables nested or relative transformations.

- **Geometry-Specific Parameters**: Additional attributes tailored to the surface type. For example:

  - **Spherical Geometry**: Radius of curvature.
  - **Conic Geometry**: Conic constant and radius of curvature.
  - **Aspheric Geometry**: Coefficients for higher-order terms (e.g., polynomial aspheric terms).
  - **Custom Geometries**: User-defined parameters. This flexibility allows for a wide range of surface shapes, including freeforms.

.. note::
    Geometries are defined in their local coordinate systems. They are then transformed based on their `CoordinateSystem` argument, which specifies their position and orientation in the global coordinate space.

Intersection and Normal Computation
-----------------------------------

Geometries provide methods for:

1. **Finding the Intersection Point**:

   - For simple shapes (e.g., planes, spheres, conics), this is computed using closed-form equations.
   - For complex shapes (e.g., polynomial XY, Chebyshev surfaces, or aspheres), iterative methods like Newton-Raphson are used to find the intersection.
   - A **Newton-Raphson Base Class** is included for geometries that require iterative solutions. Simply subclass this base class and implement the required methods for your geometry.

2. **Computing Surface Normals**:

   - Normals are derived from the mathematical description of the geometry and are essential for determining the ray's direction after refraction or reflection.

Supported Geometry Types
------------------------

Optiland includes a wide range of built-in geometries:

- **Planes**: Flat surfaces with infinite or finite extent.
- **Spheres and Conics**: Defined by radius of curvature and conic constant.
- **Even Aspheres**: Described by polynomial terms for deviations from a sphere.
- **Odd Aspheres**: Similar to even aspheres but with additional terms for odd powers.
- **Polynomial XY and Chebyshev Surfaces**: Useful for advanced freeform optical systems.
- **Toridal Surfaces**: Defined by two radii of curvature, allowing for toroidal shapes.
- **Zernike Surfaces**: Represented by Zernike polynomials. For a detailed mathematical description of the Zernike geometry, see the `Zernike Geometry Mathematics Reference <https://github.com/HarrisonKramer/optiland/blob/master/docs/references/zernike_description.md>`_.
- **Custom Geometries**: Users can easily extend the framework by subclassing the `BaseGeometry` (analytical geometries) or `NewtonRaphsonGeometry` (iterative geometries) classes.

Extensibility
-------------

Adding new geometries is straightforward:

1. Subclass the `BaseGeometry` base class or the `NewtonRaphsonGeometry` class.
2. For `BaseGeometry` implement the `distance(rays)`, `sag(x, y)` and `surface_normal(rays)` methods. For `NewtonRaphsonGeometry` implement the `sag(x, y)` and `_surface_normal(x, y)` methods.
3. Optionally, define additional parameters in the constructor for the geometry's specific shape.


.. tip::
   See the **Surface Overview** section for how geometries integrate with surfaces.
