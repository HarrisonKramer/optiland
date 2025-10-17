.. _interaction_models:

Interaction Models
==================

Interaction models define how rays interact with a surface. Each `Surface` object has an `InteractionModel` that determines whether a ray is refracted, reflected, or diffracted.

BaseInteractionModel
--------------------

The `BaseInteractionModel` is an abstract base class that defines the interface for all interaction models. It has two main methods:

- `interact_real_rays(rays)`: Interacts with real rays.
- `interact_paraxial_rays(rays)`: Interacts with paraxial rays.

RefractiveReflectiveModel
-------------------------

The `RefractiveReflectiveModel` is the most common interaction model. It handles both refraction and reflection based on the `is_reflective` flag.

ThinLensInteractionModel
------------------------

The `ThinLensInteractionModel` is used for paraxial surfaces. It simplifies the surface to an ideal thin lens with a given focal length.

DiffractiveInteractionModel
-------------------------

The `DiffractiveInteractionModel` is used for surfaces with physical diffraction gratings. It calculates the new direction of the ray based on the grating equation.

PhaseInteractionModel
---------------------

The `PhaseInteractionModel` provides a flexible way to define the phase of a surface. It uses a `BasePhase` object to calculate the phase at each point on the surface. This allows for the creation of various optical elements, such as gratings, lenses with custom phase profiles, and more.

A common `BasePhase` subclass is `GratingPhase`, which models a simple diffraction grating. For more complex phase profiles, you can create your own subclass of `BasePhase`.
