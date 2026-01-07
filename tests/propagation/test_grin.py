"""Unit tests for the GRIN propagation model."""
import pytest

import optiland.backend as be
from optiland.materials import GradientMaterial
from optiland.propagation.grin import GRINPropagation
from optiland.rays.real_rays import RealRays


def test_grin_material_initialization():
    """Verify that GradientMaterial can be initialized."""
    material = GradientMaterial(n0=1.5, nr2=0.1, nr4=0.01)
    assert material.n0 == 1.5
    assert material.nr2 == 0.1
    assert material.nr4 == 0.01


def test_grin_material_refractive_index():
    """Verify that GradientMaterial calculates refractive index correctly."""
    material = GradientMaterial(n0=1.5, nr2=0.1, nz1=0.05)

    # At origin, should be n0
    n = material.n(0.5, x=0.0, y=0.0, z=0.0)
    assert be.allclose(n, 1.5)

    # At x=1, y=0, z=0, r^2=1, should be n0 + nr2*1
    n = material.n(0.5, x=1.0, y=0.0, z=0.0)
    assert be.allclose(n, 1.6)

    # At x=0, y=0, z=1, should be n0 + nz1*1
    n = material.n(0.5, x=0.0, y=0.0, z=1.0)
    assert be.allclose(n, 1.55)


def test_grin_material_get_index_and_gradient():
    """Verify that GradientMaterial calculates gradient correctly."""
    material = GradientMaterial(n0=1.5, nr2=0.1, nz1=0.05)

    n, dn_dx, dn_dy, dn_dz = material.get_index_and_gradient(1.0, 0.0, 0.0, 0.5)

    # n at x=1, y=0, z=0: n0 + nr2*1^2 = 1.6
    assert be.allclose(n, 1.6)

    # dn/dx at x=1: 2*nr2*x = 0.2
    assert be.allclose(dn_dx, 0.2)

    # dn/dy at y=0: 2*nr2*y = 0.0
    assert be.allclose(dn_dy, 0.0)

    # dn/dz: nz1 = 0.05
    assert be.allclose(dn_dz, 0.05)


def test_grin_propagation_initialization():
    """Verify that GRINPropagation can be initialized."""
    material = GradientMaterial(n0=1.5, nr2=0.1)
    model = GRINPropagation(material)
    assert model.material is material
    assert model.step_size == 0.01


def test_grin_propagation_straight_line():
    """Test GRIN propagation with uniform refractive index (no gradient)."""
    # Create a uniform material (no gradient)
    material = GradientMaterial(n0=1.5, nr2=0.0, nr4=0.0, nr6=0.0,
                                nz1=0.0, nz2=0.0, nz3=0.0)

    # Create rays propagating in z-direction
    rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[0.5]
    )

    # Propagate through the material
    model = GRINPropagation(material)
    model.propagate(rays, t=5.0)

    # Ray should have moved 5 mm in z-direction
    assert be.allclose(rays.z[0], 5.0, atol=1e-3)
    # x and y should remain unchanged
    assert be.allclose(rays.x[0], 0.0, atol=1e-6)
    assert be.allclose(rays.y[0], 0.0, atol=1e-6)
    # Direction should still be in z-direction
    assert be.allclose(rays.L[0], 0.0, atol=1e-6)
    assert be.allclose(rays.M[0], 0.0, atol=1e-6)
    assert be.allclose(rays.N[0], 1.0, atol=1e-6)


def test_grin_propagation_radial_gradient():
    """Test GRIN propagation with radial gradient."""
    # Create a material with radial gradient
    # Positive nr2 means refractive index increases with radius
    # This causes rays to bend toward higher index (outward, away from axis)
    material = GradientMaterial(n0=1.5, nr2=0.01)

    # Create ray starting at x=0.5, propagating in z-direction
    rays = RealRays(
        x=[0.5], y=[0.0], z=[0.0],
        L=[0.0], M=[0.0], N=[1.0],
        intensity=[1.0], wavelength=[0.5]
    )

    # Propagate through the material
    model = GRINPropagation(material)
    model.propagate(rays, t=10.0)

    # With positive nr2, ray should bend away from the axis (x increases)
    # because the refractive index is higher away from the center
    assert rays.x[0] > 0.5
    # z should be approximately 10
    assert be.allclose(rays.z[0], 10.0, atol=1e-3)


def test_grin_propagation_multiple_rays():
    """Test GRIN propagation with multiple rays."""
    material = GradientMaterial(n0=1.5, nr2=0.005)

    # Create multiple rays at different x positions
    rays = RealRays(
        x=[0.1, 0.3, 0.5], y=[0.0, 0.0, 0.0],
        z=[0.0, 0.0, 0.0],
        L=[0.0, 0.0, 0.0], M=[0.0, 0.0, 0.0],
        N=[1.0, 1.0, 1.0],
        intensity=[1.0, 1.0, 1.0],
        wavelength=[0.5, 0.5, 0.5]
    )

    model = GRINPropagation(material)
    model.propagate(rays, t=5.0)

    # All rays should have propagated
    assert be.all(rays.z > 4.9)

    # Check that GRINPropagation has the material's propagation model
    assert isinstance(material.propagation_model, GRINPropagation)


def test_grin_material_serialization():
    """Test that GradientMaterial can be serialized and deserialized."""
    material = GradientMaterial(n0=1.6, nr2=0.02, nr4=0.001, nz1=0.01)

    # Serialize
    data = material.to_dict()

    # Deserialize
    material2 = GradientMaterial.from_dict(data)

    # Check that parameters match
    assert material2.n0 == material.n0
    assert material2.nr2 == material.nr2
    assert material2.nr4 == material.nr4
    assert material2.nz1 == material.nz1


def test_grin_propagation_serialization():
    """Test that GRINPropagation can be serialized and deserialized."""
    material = GradientMaterial(n0=1.5, nr2=0.1)

    # Serialize propagation model
    data = material.propagation_model.to_dict()

    # Deserialize
    model2 = GRINPropagation.from_dict(data, material)

    # Check that it's a valid GRINPropagation instance
    assert isinstance(model2, GRINPropagation)
    assert model2.material is material


def test_grin_axial_gradient_reflection_effect():
    """Test GRIN propagation with axial gradient creating gradual reflection.

    When the refractive index increases in the z-direction, obliquely incident
    rays bend toward the normal (z-axis), creating a gradual "reflection"
    or turning effect.
    """
    # Create a material with strong axial gradient
    # n(z) = n0 + nz1*z
    # nz1 > 0 means refractive index increases with z
    n0 = 1.5
    nz1 = 0.05  # Strong gradient: n increases from 1.5 at z=0 to 2.5 at z=20
    material = GradientMaterial(n0=n0, nz1=nz1)

    # Create a ray starting at origin with oblique angle
    # Ray direction: L=0.3, M=0, N=sqrt(1-0.3^2) ≈ 0.954
    # This means the ray is at ~16.7° from the z-axis
    initial_angle = 0.3  # sin(angle) ≈ 0.3
    L_init = initial_angle
    N_init = be.sqrt(1 - initial_angle**2)

    rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[L_init], M=[0.0], N=[N_init],
        intensity=[1.0], wavelength=[0.5]
    )

    # Store initial direction
    L_initial = rays.L[0]
    N_initial = rays.N[0]

    # Propagate through the axial GRIN medium
    thickness = 20.0
    model = GRINPropagation(material)
    model.propagate(rays, t=thickness)

    # As n increases with z, the ray should bend toward the normal (z-axis)
    # This means L should decrease and N should increase
    assert rays.L[0] < L_initial, "Ray should bend toward z-axis (L decreases)"
    assert rays.N[0] > N_initial, "Ray should bend toward z-axis (N increases)"

    # The ray should still be propagating forward (z should increase)
    assert rays.z[0] > 0

    # Verify refractive index at different positions
    n_start = material.n(0.5, x=0.0, y=0.0, z=0.0)
    n_end = material.n(0.5, x=0.0, y=0.0, z=thickness)

    assert be.allclose(n_start, n0, atol=1e-6)
    assert be.allclose(n_end, n0 + nz1 * thickness, atol=1e-6)

    # The stronger the gradient, the more the ray bends
    # Calculate the bending angle
    final_L = rays.L[0]
    final_N = rays.N[0]
    bending_amount = L_initial - final_L

    assert bending_amount > 0, "Ray should have bent toward z-axis"

    # With this strong gradient, we expect significant bending
    # (at least 10% reduction in L component)
    assert bending_amount > 0.1 * L_initial, \
        f"Expected significant bending, got {bending_amount:.4f}"


def test_grin_axial_gradient_strong_reflection():
    """Test strong axial gradient that can almost reflect the ray.

    With a very strong gradient in n, the ray can bend so much that it
    approaches a turning point or even reflects back.
    """
    # Create a material with very strong axial gradient
    n0 = 1.3
    nz1 = 0.1  # Very strong gradient

    material = GradientMaterial(n0=n0, nz1=nz1)

    # Create a ray with significant initial angle
    L_init = 0.5  # ~30° from z-axis
    N_init = be.sqrt(1 - L_init**2)

    rays = RealRays(
        x=[0.0], y=[0.0], z=[0.0],
        L=[L_init], M=[0.0], N=[N_init],
        intensity=[1.0], wavelength=[0.5]
    )

    # Propagate through medium
    thickness = 10.0
    model = GRINPropagation(material)
    model.propagate(rays, t=thickness)

    # With such a strong gradient, the ray should bend significantly
    # toward the z-axis
    assert rays.L[0] < L_init

    # Check that the ray reached the target distance
    assert rays.z[0] > thickness * 0.95  # Allow small numerical error

