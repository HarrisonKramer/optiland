import optiland.backend as be
import pytest

from optiland.optimization import operand
from optiland.samples.telescopes import HubbleTelescope
from .utils import assert_allclose

from optiland.optic import Optic
from optiland.surfaces import StandardSurface
from optiland.geometries import Plane
from optiland.materials import IdealMaterial
from optiland.wavelength import Wavelengths
from optiland.fields import Fields
from optiland.optimization.operand import RayOperand # Direct import for clarity


@pytest.fixture
def hubble():
    return HubbleTelescope()


class TestParaxialOperand:
    def test_f1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.f1(hubble), -57600.08099840297)

    def test_f2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.f2(hubble), 57600.080998403595)

    def test_F1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.F1(hubble), -471891.9369498712)

    def test_F2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.F2(hubble), 0.016864237529848742)

    def test_P1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.P1(hubble), -414291.8559514682)

    def test_P2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.P2(hubble), -57600.06413416607)

    def test_N1(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.N1(hubble), -414291.8559514676)

    def test_N2(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.N2(hubble), -57600.06413416545)

    def test_EPL(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.EPL(hubble), 4910.01016)

    def test_EPD(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.EPD(hubble), 2400)

    def test_XPL(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.XPL(hubble), -6958.363551642134)

    def test_XPD(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.XPD(hubble), 289.93210961932573)

    def test_magnification(self, set_test_backend, hubble):
        assert_allclose(operand.ParaxialOperand.magnification(hubble), 0.0)


class TestAberrationOperand:
    def test_seidel(self, set_test_backend, hubble):
        assert_allclose(
            operand.AberrationOperand.seidels(hubble, 0),
            0.0014539022855417389,
        )

    def test_TSC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TSC(hubble, 2), 0.0)

    def test_SC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.SC(hubble, 2), 0.0)

    def test_CC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.CC(hubble, 2), 0.0)

    def test_TCC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TCC(hubble, 1), 0.0)

    def test_TAC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAC(hubble, 1), 0.0)

    def test_AC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.AC(hubble, 1), 0.0)

    def test_TPC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TPC(hubble, 1), 0.0)

    def test_PC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.PC(hubble, 1), 0.0)

    def test_DC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.DC(hubble, 1), 0.0)

    def test_TAchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAchC(hubble, 1), 0.0)

    def test_LchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.LchC(hubble, 1), 0.0)

    def test_TchC(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TchC(hubble, 1), 0.0)

    def test_TSC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TSC_sum(hubble), 0.0)

    def test_SC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.SC_sum(hubble), 0.0)

    def test_CC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.CC_sum(hubble), 0.0)

    def test_TCC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TCC_sum(hubble), 0.0)

    def test_TAC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAC_sum(hubble), 0.0)

    def test_AC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.AC_sum(hubble), 0.0)

    def test_TPC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TPC_sum(hubble), 0.0)

    def test_PC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.PC_sum(hubble), 0.0)

    def test_DC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.DC_sum(hubble), 0.03489370392123652)

    def test_TAchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TAchC_sum(hubble), 0.0)

    def test_LchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.LchC_sum(hubble), 0.0)

    def test_TchC_sum(self, set_test_backend, hubble):
        assert_allclose(operand.AberrationOperand.TchC_sum(hubble), 0.0)


class TestRayOperand:
    def test_x_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 1.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.x_intercept(**data), 150.42338010762842)

    def test_y_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.y_intercept(**data), 150.42338010762842)

    def test_z_intercept(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.z_intercept(**data), 6347.146837237045)

    def test_x_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 1.0,
            "Hy": 0.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.x_intercept_lcs(**data),
            150.42338010762842,
        )

    def test_y_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.y_intercept_lcs(**data),
            150.42338010762842,
        )

    def test_z_intercept_lcs(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(
            operand.RayOperand.z_intercept_lcs(**data),
            -18.062712762936826,
        )  # Because Hubble's image is curved, otherwise it would be 0

    def test_L(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.L(**data), 0.0)

    def test_M(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.M(**data), 0.021668820646914833)

    def test_N(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "Px": 0.0,
            "Py": 0.0,
            "wavelength": 0.55,
        }
        assert_allclose(operand.RayOperand.N(**data), 0.9997652035411982)

    def test_rms_spot_size(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 100,
        }
        assert_allclose(
            operand.RayOperand.rms_spot_size(**data),
            0.025626727777956947,
        )

    def test_rms_spot_size_all_wave(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "surface_number": -1,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": "all",
            "num_rays": 100,
        }
        assert_allclose(
            operand.RayOperand.rms_spot_size(**data),
            0.025626727777956947,
        )

    def test_opd_diff(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 3,
        }
        assert_allclose(operand.RayOperand.OPD_difference(**data), 0.0732334055041471)

    def test_opd_diff_on_axis(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 0.0,
            "wavelength": 0.55,
            "num_rays": 3,
        }
        assert_allclose(
            operand.RayOperand.OPD_difference(**data),
            0.0013295118803701674,
        )

    def test_opd_diff_new_dist(self, set_test_backend, hubble):
        data = {
            "optic": hubble,
            "Hx": 0.0,
            "Hy": 1.0,
            "wavelength": 0.55,
            "num_rays": 100,
            "distribution": "hexapolar",
        }
        assert_allclose(operand.RayOperand.OPD_difference(**data), 0.2211995620762635)

    def create_test_optic(self):
        optic = Optic()
        optic.wavelengths = Wavelengths([0.55])
        # Set a known entrance pupil diameter for predictable Px,Py to angle conversion if needed.
        # However, for this test, field coords (Hx,Hy) are more direct for yB positions.
        # And for Line A, Px,Py will be used to generate mA, nA whose exact values are taken from trace.
        optic.EPD = 20.0 # Example EPD, helps interpret Px,Py if they were used for position.
                        # Actual angles M, N are taken from trace, so EPD value is not critical for test logic.

        optic.fields = Fields()
        # Add a default field for tracing, Optic requires at least one.
        # Max field of 1.0 means Hx=1 or Hy=1 corresponds to an angle of 1 radian here.
        # This is a simplification for test predictions. Actual system might use degrees.
        optic.fields.add_field(angle_x=0, angle_y=0, weight=1, max_angle_x=1.0, max_angle_y=1.0)


        # Object surface (idx 0)
        s0 = StandardSurface(geometry=Plane(), material=IdealMaterial(1.0))
        optic.surface_group.add_surface(s0) # Object surface is at z=0
        # Surface 1 (idx 1) - Line A ray data taken AFTER this surface
        s1 = StandardSurface(geometry=Plane(), material=IdealMaterial(1.0))
        s1.thickness = 10.0 # s1 is at z=10
        optic.surface_group.add_surface(s1)
        # Surface 2 (idx 2) - Point B ray data taken AT this surface
        s2 = StandardSurface(geometry=Plane(), material=IdealMaterial(1.0))
        s2.thickness = 10.0 # s2 is at z=20
        optic.surface_group.add_surface(s2)
        # Image surface (idx 3)
        s3 = StandardSurface(geometry=Plane(), is_image_surface=True) # Image surface is at z=20
        optic.surface_group.add_surface(s3)
        optic.update()
        return optic

    def test_clearance(self, set_test_backend):
        set_test_backend()
        optic = self.create_test_optic()
        wavelength = 0.55

        # Test Case 1: Point B on +Y side of Line A (nA > 0)
        # Line A: from surface 1, on-axis ray (Hx=0,Hy=0; Px=0,Py=0).
        # This ray travels along Z. yA=0, zA=10 (global for surface 1), mA=0, nA=1.
        # Point B: at surface 2, ray starting at Hx=0, Hy=0.1 (field y-angle = 0.1 rad).
        # This ray will hit surface 2 at yB = 0.1 * (z_s2 - z_obj) = 0.1 * 20 = 2.0.
        # optic.trace_generic(0.0, 0.1, 0.0, 0.0, wavelength)
        # yB_expected_case1 = optic.surface_group.y[2, 0] # Should be 2.0

        dist1 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, 0.0),
            point_ray_surface_idx=2,
            point_ray_field_coords=(0.0, 0.1), # Hy = 0.1 rad
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength
        )
        # Expected: Line A (yA=0, zA=10, mA=0, nA=1). Point B (yB=2.0, zB=20).
        # dist = (1 * (2.0 - 0) - 0 * (20 - 10)) / 1 = 2.0
        assert be.isclose(dist1, 2.0, atol=1e-7), f"Test Case 1 Failed: Expected 2.0, Got {dist1}"

        # Test Case 2: Point B on -Y side of Line A (nA > 0)
        # Point B: at surface 2, ray Hx=0, Hy=-0.1. yB = -0.1 * 20 = -2.0.
        dist2 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, 0.0),
            point_ray_surface_idx=2,
            point_ray_field_coords=(0.0, -0.1), # Hy = -0.1 rad
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength
        )
        # Expected: yB = -2.0. dist = -2.0
        assert be.isclose(dist2, -2.0, atol=1e-7), f"Test Case 2 Failed: Expected -2.0, Got {dist2}"

        # Test Case 3: Point B on Line A (nA > 0)
        # Point B: at surface 2, ray Hx=0, Hy=0. yB = 0.
        dist3 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, 0.0),
            point_ray_surface_idx=2,
            point_ray_field_coords=(0.0, 0.0),
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength
        )
        # Expected: yB = 0.0. dist = 0.0
        assert be.isclose(dist3, 0.0, atol=1e-7), f"Test Case 3 Failed: Expected 0.0, Got {dist3}"

        # Test Case 4: Line A with non-zero mA (tilted line, nA > 0)
        # Line A: from surf 1. Field (Hx,Hy)=(0,0). Pupil (Px,Py)=(0,0.1).
        # This means on-axis object point, ray aimed at y_EP = 0.1 * EPD/2 on Entrance Pupil.
        # EPD = 20, so y_EP = 0.1 * 10 = 1.0.
        # Ray from (0,0,0) to (0,1,z_EP). If EP at object (z_EP=0), slope M_obj = 1/sqrt(0^2+1^2+0^2) is not how M is defined.
        # M is dy/ds. If ray goes from (0,0,0) to (0,1,0) this is not a propagating ray.
        # For a ray starting at object (0,0,0) and passing through y_EP=1 at z_EP=0 (if EP is at object):
        # The initial direction cosines M, N before hitting surf 1:
        # optic.trace_generic will trace this.
        # M_before_s1 = Py (if object at inf and Py is angle)
        # M_before_s1 = y_EP / distance_to_first_surface (approx if EP far from object)
        # Let's get traced values for Line A after surface 1:
        optic.trace_generic(Hx=0.0, Hy=0.0, Px=0.0, Py=0.1, wavelength=wavelength)
        yA_t4 = optic.surface_group.y[1, 0] # y-coordinate on surface 1 after propagation from object
        zA_t4 = optic.surface_group.z[1, 0] # z-coordinate of surface 1 (10.0)
        mA_t4 = optic.surface_group.M[1, 0] # M direction cosine after surface 1
        nA_t4 = optic.surface_group.N[1, 0] # N direction cosine after surface 1
        # With EPD=20, Py=0.1: ray goes from (0,0,0) through (0,1,0) effectively (if EP at obj).
        # Slope y/z = 1/10 to surface 1. So yA_t4 = 1.0.
        # True M = (yA_t4-0)/sqrt(yA_t4^2 + (zA_t4-0)^2) = 1/sqrt(1+100) = 1/sqrt(101) approx 0.0995
        # True N = zA_t4/sqrt(101) = 10/sqrt(101) approx 0.995
        # The values from trace_generic are direction cosines after refraction, which is just pass-through for plane+IdealMaterial.
        # So, yA_t4 = 1.0, zA_t4 = 10.0. mA_t4 approx 0.0995, nA_t4 approx 0.995.

        # Point B: at surface 2, field (0,0), pupil (0,0) (chief ray for on-axis field)
        # yB_t4 = 0.0, zB_t4 = 20.0
        yB_val_case4 = 0.0
        zB_val_case4 = 20.0

        dist4 = RayOperand.clearance(
            optic=optic,
            line_ray_surface_idx=1,
            line_ray_field_coords=(0.0, 0.0),
            line_ray_pupil_coords=(0.0, 0.1),
            point_ray_surface_idx=2,
            point_ray_field_coords=(0.0, 0.0),
            point_ray_pupil_coords=(0.0, 0.0),
            wavelength=wavelength
        )

        expected_dist4_num = nA_t4 * (yB_val_case4 - yA_t4) - mA_t4 * (zB_val_case4 - zA_t4)
        expected_dist4_den = be.sqrt(mA_t4**2 + nA_t4**2)
        # Denominator should be close to 1 if L,M,N are normalized (L^2+M^2+N^2=1 and L=0 here)
        # So expected_dist4_den = sqrt(M^2+N^2). If L is non-zero, then it's sqrt(M^2+N^2) from formula, not sqrt(1-L^2).
        # The formula for d uses mA, nA which are M, N of Line A in YZ plane.
        # This is correct.
        
        # Using traced values:
        # yA_t4 = 1.0, zA_t4 = 10.0
        # mA_t4 = 0.09950371902099892, nA_t4 = 0.9950371902099892 (from EPD=20 setup)
        # yB_val_case4 = 0.0, zB_val_case4 = 20.0
        # num = 0.99503719 * (0.0 - 1.0) - 0.099503719 * (20.0 - 10.0)
        #     = -0.99503719 - 0.099503719 * 10
        #     = -0.99503719 - 0.99503719 = -1.99007438
        # den = be.sqrt(0.099503719**2 + 0.99503719**2) = be.sqrt(0.00990099 + 0.990099) = be.sqrt(1.0) = 1.0
        # (This is because L_t4 from this trace is essentially 0)
        # expected_dist4 = -1.99007438 / 1.0 = -1.99007438
        expected_dist4 = (nA_t4 * (yB_val_case4 - yA_t4) - mA_t4 * (zB_val_case4 - zA_t4)) / be.sqrt(mA_t4**2 + nA_t4**2)
        assert be.isclose(dist4, expected_dist4, atol=1e-7), f"Test Case 4 Failed: Expected {expected_dist4}, Got {dist4}"

        # Test Case 5: Sign convention with nA < 0
        # This is difficult to set up with the current simple forward-propagating optic
        # without specific surfaces (like mirrors) or direct mocking of ray data.
        # The formula includes `if nA < 0: d = -d`, which handles the sign flip.

        # Test Case 6: Denominator is zero (mA=0, nA=0)
        # This implies the line ray has no propagation component in the YZ plane (i.e., it's purely along X).
        # This is very hard to achieve reliably via standard tracing in this simple system.
        # The clearance function has a built-in check: `if be.abs(denominator) < epsilon: d = 0.0`
        # This handles division by zero. A direct test of this condition would require mocking
        # mA and nA values to be zero, or a specialized optical system.
        # For now, we rely on this internal safeguard.
        pass


class TestOperand:
    def test_get_value(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=1,
            weight=1,
            input_data=input_data,
        )
        assert_allclose(op.value, 57600.080998403595)

    def test_invalid_operand(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f3",
            target=1,
            weight=1,
            input_data=input_data,
        )
        with pytest.raises(ValueError):
            op.value

    def test_delta(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=5000,
            weight=1,
            input_data=input_data,
        )
        assert_allclose(op.delta(), 52600.080998403595)

    def test_fun(self, set_test_backend, hubble):
        input_data = {"optic": hubble}
        op = operand.Operand(
            operand_type="f2",
            target=1e5,
            weight=1.5,
            input_data=input_data,
        )
        assert_allclose(op.fun(), -63599.87850239461)

    def test_reregister_operand(self, set_test_backend):
        try:
            operand.operand_registry._registry.pop("f3")
        except KeyError:
            pass

        operand.operand_registry.register("f3", lambda x: x)
        with pytest.raises(ValueError):
            operand.operand_registry.register("f3", lambda x: x)

    def test_contains(self, set_test_backend):
        assert "f1" in operand.operand_registry
        assert "invalid" not in operand.operand_registry

    def test_repr(self, set_test_backend):
        assert isinstance(repr(operand.operand_registry), str)

    def test_invalid_bounds(self, set_test_backend):
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0)

    def test_invalid_bounds_and_target(self, set_test_backend):
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0, target=1)

    def test_delta_ineq(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(
            operand_type="f2",
            min_val=1e6,
            max_val=1e11,
            weight=1,
            input_data=input_data,
        )
        hubble = HubbleTelescope()
        f2 = hubble.paraxial.f2()

        # check on the lower bound
        assert_allclose(op.delta_ineq(), 1e6 - f2)

        op = operand.Operand(
            operand_type="f2",
            min_val=0,
            max_val=1,
            weight=1,
            input_data=input_data,
        )

        # check on the upper bound
        assert_allclose(op.delta_ineq(), f2 - 1)

    def test_delta_ineq_no_bounds(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(operand_type="f2", weight=1, input_data=input_data)
        assert_allclose(op.delta_ineq(), 0)

    def test_delta_with_bounds(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(
            operand_type="f2",
            min_val=1e6,
            max_val=1e11,
            weight=1,
            input_data=input_data,
        )
        hubble = HubbleTelescope()
        f2 = hubble.paraxial.f2()

        # check on the lower bound
        assert_allclose(op.delta(), 1e6 - f2)

        op = operand.Operand(
            operand_type="f2",
            min_val=0,
            max_val=1,
            weight=1,
            input_data=input_data,
        )

        # check on the upper bound
        assert_allclose(op.delta(), f2 - 1)

    def test_invalid_delta(self, set_test_backend):
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(operand_type="f2", input_data=input_data)
        # manually remove target to make the operand invalid
        op.target = None
        with pytest.raises(ValueError):
            op.delta()


class TestOperandManager:
    def test_add(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert len(manager) == 1

    def test_clear(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.clear()
        assert len(manager) == 0

    def test_iter(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.add("f2", 1)
        for op in manager:
            assert isinstance(op, operand.Operand)

    def test_getitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert isinstance(manager[0], operand.Operand)

    def test_setitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager[0] = operand.Operand(
            operand_type="f2",
            target=1,
            weight=1,
            input_data={},
        )
        assert len(manager) == 1

    def test_len(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.add("f2", 1)
        assert len(manager) == 2

    def test_getitem_error(self, set_test_backend):
        manager = operand.OperandManager()
        with pytest.raises(IndexError):
            manager[0]

    def test_setitem_invalid_type(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        with pytest.raises(ValueError):
            manager[0] = 1

    def test_delitem(self, set_test_backend):
        manager = operand.OperandManager()
        manager.add("f1", 1)
        del manager[0]
        assert len(manager) == 0
