.. _ray_overview:

Ray Tracing Framework
=====================

The ray tracing framework in Optiland simulates light propagation through an optical system. It handles various types of rays,
supports flexible ray generation, and traces rays across surfaces in sequential optical systems.

Ray Representation
------------------
Rays are the fundamental elements of the ray tracing process. A ray consists of the following attributes:

- **Origin (x, y, z)**: The starting point of the ray.
- **Direction (L, M, N)**: The ray's unit vector indicating its direction.
- **Wavelength**: The wavelength of the ray in the system.
- **Intensity**: The ray's relative intensity.
- **Optical Path Length (OPL)**: The accumulated path the ray has traveled, weighted by refractive index.
- **Polarization Matrix (optional)**: A 3x3 matrix representing the transformation of a ray's initial electric field into its final state.

.. note::
  In Optiland, all ray attributes are defined as NumPy arrays for efficient computation.

Ray Types
---------
Optiland supports three types of rays:

- **Real Rays**: Standard rays with full 3D attributes, used for detailed tracing.
- **Paraxial Rays**: Simplified rays using only height (`y`), angle (`u`), wavelength, and starting position (`z`). These are used for first-order system analysis.
- **Polarized Rays**: Rays with additional polarization information, allowing simulations involving birefringent materials or coatings.

Ray Generation
--------------
The ray tracing framework uses a **Ray Generator** to produce rays for tracing. The generator operates based on user-defined parameters, including:

- Field points
- Wavelengths
- Aperture sampling (e.g., uniform grid, random, etc.)
- Polarization state
- System properties (e.g., F/#, NA, telecentricity)

Generated rays are passed to the **Surface Group** for tracing through the optical system. Each `Optic` instance has both a ray generator and a surface group
specific to that system.

Ray Tracing
-----------
For real rays, the ray tracing process is managed by the **RealRayTracer**. The ray tracer is responsible for:

- Generating the appropriate rays based on the selected ray generator.
- Ray tracing input validation.
- Propagating rays through the surface group.

For paraxial rays, the **ParaxialRayTracer** is used. This tracer is similar in form to the RealRayTracer but uses simplified equations for faster computations.

Tracing Process
---------------
Ray tracing is performed by sequentially propagating rays across the surfaces in the system:

1. Rays are converted into the local coordinate system of the current surface.
2. Rays intersections with the surface are identified and the ray propagates to the intersection point.
3. If the surface has a physica aperture, rays may be clipped (intensity set to zero) if they fall outside this aperture.
4. The rays interact with the surface, modifying their direction, intensity, polarization matrix, and other attributes.
5. Ray are transformed back to the global coordinate system.
6. Ray information (intersection points, intensities, etc.) is recorded on the surface for later analysis and visualization.
7. The process is repeated for each surface in the system.

Paraxial rays follow a similar process but use simplified equations for faster computations.

.. note::
   For details on how rays interact with surfaces, see the :ref:`surface_overview` section.

Extensibility
-------------

The framework is designed to be extensible:

- New ray types can be added by subclassing the `BaseRays` base class.
- Custom ray generators can be implemented by following the existing interface.
- Additional tracing logic can be integrated into the surface `trace` method for specialized applications.

For a practical example of ray tracing, see the :ref:`getting_started` section.
