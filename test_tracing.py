from optiland.optic import Optic
from optiland.materials import IdealMaterial, Material
from optiland.coordinate_system import CoordinateSystem
from optiland import geometries
import numpy as np
from optiland.rays import RealRays
import optiland.backend as be


def basic_toroid_geometry():
    """Provides a basic ToroidalGeometry instance for testing"""
    cs = CoordinateSystem(x=0, y=0, z=0)
    radius_rotation = 100.0  # R (X-Z radius)
    radius_yz = 50.0  # R_y (Y-Z radius)
    conic = -0.5  # k_yz (YZ conic)
    coeffs_poly_y = [1e-5]
    return geometries.ToroidalGeometry(
        coordinate_system=cs,
        radius_rotation=radius_rotation,
        radius_yz=radius_yz,
        conic=conic,
        coeffs_poly_y=coeffs_poly_y,
    )


lens = Optic()
lens.add_surface(index=0, thickness=np.inf)
lens.add_surface(
    index=1,
    surface_type="toroidal",
    thickness=5.0,
    material=IdealMaterial(n=1.5, k=0),
    is_stop=True,
    radius=100.0,
    radius_y=50.0,
    conic=-0.5,
    toroidal_coeffs_poly_y=[0.05, 0.0002],
)
lens.add_surface(index=2, thickness=10.0, material="air")
lens.add_surface(index=3)

lens.set_aperture(aperture_type="EPD", value=10.0)
lens.add_wavelength(value=0.550, is_primary=True)
lens.set_field_type("angle")
lens.add_field(y=0)

lens.draw()

num_rays = 5  # Number of rays per fan
wavelength = 0.550
z_start = 0.0

# --- Tangential (Y) Fan Test ---
y_coords = be.linspace(-5.0, 5.0, num_rays)
x_in_yfan = be.zeros(num_rays)
y_in_yfan = y_coords
z_in_yfan = be.array([z_start] * num_rays)
L_in_yfan = be.zeros(num_rays)
M_in_yfan = be.zeros(num_rays)
N_in_yfan = be.ones(num_rays)
intensity_yfan = be.ones(num_rays)
rays_in_yfan = RealRays(
    x=x_in_yfan,
    y=y_in_yfan,
    z=z_in_yfan,
    L=L_in_yfan,
    M=M_in_yfan,
    N=N_in_yfan,
    wavelength=wavelength,
    intensity=intensity_yfan,
)

# Trace Y-Fan Rays
rays_out_yfan = lens.surface_group.trace(rays_in_yfan)

print("Y-Fan Rays:")
print(rays_out_yfan.y)
