.. _code_structure:

Code Structure
==============

This page provides a high-level overview of the `optiland` package's structure. The diagram below illustrates the relationships between the key components.

.. code-block:: text

    optiland/
    ├── optic/
    │   └── optic.py (Optic class)
    │
    ├── surfaces/
    │   ├── surface_group.py (SurfaceGroup class)
    │   └── standard_surface.py (Surface class)
    │
    ├── geometries/
    │   ├── base.py (BaseGeometry class)
    │   └── ... (various geometry implementations)
    │
    ├── materials/
    │   ├── base.py (BaseMaterial class)
    │   └── ... (various material implementations)
    │
    ├── interactions/
    │   ├── base.py (BaseInteractionModel class)
    │   └── ... (various interaction model implementations)
    │
    ├── propagation/
    │   ├── base.py (BasePropagationModel class)
    │   └── ... (various propagation model implementations)
    │
    ├── rays/
    │   ├── base.py (BaseRays class)
    │   └── ... (various ray implementations)
    │
    ├── raytrace/
    │   ├── real_ray_tracer.py (RealRayTracer class)
    │   └── paraxial_ray_tracer.py (ParaxialRayTracer class)
    │
    └── backend/
        ├── __init__.py (dynamic backend dispatcher)
        ├── numpy_backend.py
        └── torch_backend.py

Key Relationships
-----------------

- The **Optic** class is the central container for an optical system.
- It has a **SurfaceGroup**, which contains a list of **Surface** objects.
- Each **Surface** has a **Geometry**, a **Material** before and after the surface, and an **InteractionModel**.
- The **InteractionModel** defines how rays interact with the surface.
- Each **Material** has a **PropagationModel**, which defines how rays propagate through the material.
- The **RealRayTracer** and **ParaxialRayTracer** use the **SurfaceGroup** to trace **Rays** through the system.
- All numerical operations are dispatched to the active **Backend**.
