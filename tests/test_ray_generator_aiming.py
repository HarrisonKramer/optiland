import pytest
from optiland.optic.optic import Optic
from optiland.rays.ray_generator import RayGenerator
import optiland.backend as be
from optiland.rays.real_rays import RealRays

# THIS IS A COPY OF THE ORIGINAL RAY GENERATOR LOGIC FOR COMPARISON
class LegacyRayGenerator:
    """Generator class for creating rays."""

    def __init__(self, optic):
        self.optic = optic

    def generate_rays(self, Hx, Hy, Px, Py, wavelength):
        """Generates rays for tracing based on the given parameters."""
        vxf, vyf = self.optic.fields.get_vig_factor(Hx, Hy)
        vx = 1 - be.array(vxf)
        vy = 1 - be.array(vyf)
        x0, y0, z0 = self._get_ray_origins(Hx, Hy, Px, Py, vx, vy)

        if self.optic.obj_space_telecentric:
            if self.optic.field_type == "angle":
                raise ValueError(
                    'Field type cannot be "angle" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "EPD":
                raise ValueError(
                    'Aperture type cannot be "EPD" for telecentric object space.',
                )
            if self.optic.aperture.ap_type == "imageFNO":
                raise ValueError(
                    'Aperture type cannot be "imageFNO" for telecentric object space.',
                )

            sin = self.optic.aperture.value
            z = be.sqrt(1 - sin**2) / sin + z0
            z1 = be.full_like(Px, z)
            x1 = Px * vx + x0
            y1 = Py * vy + y0
        else:
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            x1 = Px * EPD * vx / 2
            y1 = Py * EPD * vy / 2
            z1 = be.full_like(Px, EPL)

        mag = be.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        L = (x1 - x0) / mag
        M = (y1 - y0) / mag
        N = (z1 - z0) / mag

        apodization = self.optic.apodization
        if apodization:
            intensity = apodization.get_intensity(Px, Py)
        else:
            intensity = be.ones_like(Px)

        wavelength_arr = be.ones_like(x1) * wavelength
        rays = RealRays(x0, y0, z0, L, M, N, be.ones_like(x0), wavelength_arr)
        rays.intensity = intensity
        rays.wavelength = wavelength_arr
        return rays

    def _get_ray_origins(self, Hx, Hy, Px, Py, vx, vy):
        obj = self.optic.object_surface
        max_field = self.optic.fields.max_field
        field_x = max_field * Hx
        field_y = max_field * Hy
        if obj.is_infinite:
            if self.optic.field_type == "object_height":
                raise ValueError(
                    'Field type cannot be "object_height" for an object at infinity.',
                )
            if self.optic.obj_space_telecentric:
                raise ValueError(
                    "Object space cannot be telecentric for an object at infinity.",
                )
            EPL = self.optic.paraxial.EPL()
            EPD = self.optic.paraxial.EPD()

            offset = self._get_starting_z_offset()

            x = -be.tan(be.radians(field_x)) * (offset + EPL)
            y = -be.tan(be.radians(field_y)) * (offset + EPL)
            z = self.optic.surface_group.positions[1] - offset

            x0 = Px * EPD / 2 * vx + x
            y0 = Py * EPD / 2 * vy + y
            z0 = be.full_like(Px, z)
        else:
            if self.optic.field_type == "object_height":
                x0 = be.array(field_x)
                y0 = be.array(field_y)
                z0 = obj.geometry.sag(x0, y0) + obj.geometry.cs.z
            elif self.optic.field_type == "angle":
                EPL = self.optic.paraxial.EPL()
                z0 = self.optic.surface_group.positions[0]
                x0 = -be.tan(be.radians(field_x)) * (EPL - z0)
                y0 = -be.tan(be.radians(field_y)) * (EPL - z0)

            if be.size(x0) == 1:
                x0 = be.full_like(Px, x0)
            if be.size(y0) == 1:
                y0 = be.full_like(Px, y0)
            if be.size(z0) == 1:
                z0 = be.full_like(Px, z0)
        return x0, y0, z0

    def _get_starting_z_offset(self):
        z = self.optic.surface_group.positions[1:-1]
        offset = self.optic.paraxial.EPD()
        return offset - be.min(z)

@pytest.fixture
def simple_optic():
    optic = Optic()
    optic.add_surface(surface_type="standard", is_infinite=True, index=0)
    optic.add_surface(radius=50, thickness=5, material="silica", index=1, surface_type="standard", is_stop=True)
    optic.add_surface(radius=-50, thickness=100, index=2, surface_type="standard")
    optic.add_surface(surface_type="standard", index=3)
    optic.set_aperture("EPD", 10)
    optic.set_field_type("angle")
    optic.add_field(y=0)
    optic.add_field(y=1)
    optic.add_wavelength(value=0.55, is_primary=True)
    optic.update()
    return optic

def test_ray_generator_aiming_consistency(simple_optic):
    """
    Tests that the refactored RayGenerator with the default ParaxialAimingStrategy
    produces the same output as the original implementation.
    """
    # GIVEN an optic and ray parameters
    optic = simple_optic
    Hx, Hy, Px, Py = 0.5, 0.5, 0.5, 0.5
    wavelength = optic.primary_wavelength

    # WHEN generating rays with the new and legacy generators
    new_generator = RayGenerator(optic)
    legacy_generator = LegacyRayGenerator(optic)

    new_rays = new_generator.generate_rays(Hx, Hy, Px, Py, wavelength)
    legacy_rays = legacy_generator.generate_rays(Hx, Hy, Px, Py, wavelength)

    # THEN the results should be identical
    assert be.allclose(new_rays.x, legacy_rays.x)
    assert be.allclose(new_rays.y, legacy_rays.y)
    assert be.allclose(new_rays.z, legacy_rays.z)
    assert be.allclose(new_rays.L, legacy_rays.L)
    assert be.allclose(new_rays.M, legacy_rays.M)
    assert be.allclose(new_rays.N, legacy_rays.N)
    assert be.allclose(new_rays.intensity, legacy_rays.intensity)
    assert be.allclose(new_rays.wavelength, legacy_rays.wavelength)
