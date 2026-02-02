
import pytest
import numpy as np
from optiland import optic
from optiland.geometries import (
    StandardGeometry,
    EvenAsphere,
    OddAsphere,
    PolynomialGeometry,
    ChebyshevPolynomialGeometry,
    ZernikePolynomialGeometry,
    GridSagGeometry,
    BiconicGeometry,
    ToroidalGeometry
)
import optiland.backend as be

def test_scale_standard():
    lens = optic.Optic()
    lens.add_surface(index=0, radius=10, thickness=5, material="Air")
    lens.add_surface(index=1) # Add a second surface so first one is not "last"
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    assert isinstance(surface.geometry, StandardGeometry)
    assert surface.geometry.radius == 20
    assert surface.thickness == 10

def test_scale_even_asphere():
    lens = optic.Optic()
    # C1 (r^4) -> index 0. C2 (r^6) -> index 1.
    coeffs = [1e-3, 1e-5] 
    lens.add_surface(index=0, radius=10, thickness=2, surface_type="even_asphere", coefficients=coeffs)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    assert isinstance(surface.geometry, EvenAsphere)
    assert surface.geometry.radius == 20
    
    s = 2.0
    expected_c0 = coeffs[0] * s**(1 - 2*(0+1))
    expected_c1 = coeffs[1] * s**(1 - 2*(1+1))
    
    assert np.isclose(surface.geometry.coefficients[0], expected_c0)
    assert np.isclose(surface.geometry.coefficients[1], expected_c1)

def test_scale_odd_asphere():
    lens = optic.Optic()
    # odd asphere: z = ... + sum(Ci * r^(i+1))
    # i=0 -> r^1.
    coeffs = [1e-2, 1e-4]
    lens.add_surface(index=0, radius=10, surface_type="odd_asphere", coefficients=coeffs)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    s = 2.0
    expected_c0 = coeffs[0] * s**(1 - (0+1))
    expected_c1 = coeffs[1] * s**(1 - (1+1))
    
    assert np.isclose(surface.geometry.coefficients[0], expected_c0)
    assert np.isclose(surface.geometry.coefficients[1], expected_c1)

def test_scale_polynomial():
    lens = optic.Optic()
    # Cij * x^i * y^j
    # i=1, j=1 -> xy. s^(1-(1+1)) = s^-1.
    coeffs = [[0, 0], [0, 1e-3]] # C11
    lens.add_surface(index=0, radius=10, surface_type="polynomial", coefficients=coeffs)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    s = 2.0
    # coeffs[1][1] is x^1 y^1
    expected = 1e-3 * s**(1 - (1+1))
    assert np.isclose(surface.geometry.coefficients[1][1], expected)

def test_scale_chebyshev():
    lens = optic.Optic()
    coeffs = [[0, 0], [0, 1e-3]]
    lens.add_surface(index=0, radius=10, surface_type="chebyshev", coefficients=coeffs, norm_x=1.0, norm_y=1.0)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    s = 2.0
    
    assert surface.geometry.norm_x == 2.0
    assert surface.geometry.norm_y == 2.0
    # Chebyshev coeffs scale linearly with s if norms are scaled
    assert surface.geometry.coefficients[1][1] == 1e-3 * s

def test_scale_zernike():
    lens = optic.Optic()
    coeffs = [0, 1e-3, 1e-4]
    lens.add_surface(index=0, radius=10, surface_type="zernike", coefficients=coeffs, norm_radius=1.0)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    s = 2.0
    
    assert surface.geometry.norm_radius == 2.0
    assert np.allclose(surface.geometry.coefficients, np.array(coeffs) * s)

def test_scale_biconic():
    lens = optic.Optic()
    lens.add_surface(index=0, radius_x=10, surface_type="biconic", radius_y=10, conic_x=0, conic_y=0) # using defaults
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    assert isinstance(surface.geometry, BiconicGeometry)
    assert surface.geometry.Rx == 20
    assert surface.geometry.Ry == 20

def test_scale_toroidal():
    lens = optic.Optic()
    coeffs = [1e-3]
    lens.add_surface(index=0, radius_x=10, surface_type="toroidal", radius_y=5, toroidal_coeffs_poly_y=coeffs)
    lens.scale_system(2.0)
    
    surface = lens.surface_group.surfaces[0]
    s = 2.0
    
    assert surface.geometry.R_rot == 20
    assert surface.geometry.R_yz == 10
    
    # Coeffs scaling: y^(2(i+1)). i=0 -> y^2. 
    # s^(1 - 2) = s^-1.
    expected = coeffs[0] * s**(1 - 2*(1))
    assert np.isclose(surface.geometry.coeffs_poly_y[0], expected)
