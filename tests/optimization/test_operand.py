# tests/optimization/test_operand.py
"""
Tests for the various optimization operands in optiland.optimization.operand.

These tests verify that the operand functions correctly calculate their respective
optical properties, which are used to construct merit functions for optimization.
"""
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.optimization import operand
from optiland.optimization.operand import RayOperand, LensOperand
from optiland.samples.telescopes import HubbleTelescope
from optiland.samples.objectives import CookeTriplet

from ..utils import assert_allclose


@pytest.fixture
def hubble():
    """Provides a HubbleTelescope instance for testing operands."""
    return HubbleTelescope()


@pytest.fixture
def cooke_triplet():
    """Provides a CookeTriplet instance for testing operands."""
    return CookeTriplet()


class TestParaxialOperand:
    """
    Tests for operands related to paraxial optical properties, such as focal
    lengths, principal plane locations, and pupil properties.
    """

    def test_f1(self, set_test_backend, hubble):
        """Tests the front focal length calculation."""
        assert_allclose(operand.ParaxialOperand.f1(hubble), -57600.08099840297)

    def test_f2(self, set_test_backend, hubble):
        """Tests the back focal length calculation."""
        assert_allclose(operand.ParaxialOperand.f2(hubble), 57600.080998403595)

    def test_F1(self, set_test_backend, hubble):
        """Tests the front focal point location calculation."""
        assert_allclose(operand.ParaxialOperand.F1(hubble), -471891.9369498712)

    def test_F2(self, set_test_backend, hubble):
        """Tests the back focal point location calculation."""
        assert_allclose(operand.ParaxialOperand.F2(hubble), 0.016864237529848742)

    def test_P1(self, set_test_backend, hubble):
        """Tests the first principal plane location calculation."""
        assert_allclose(operand.ParaxialOperand.P1(hubble), -414291.8559514682)

    def test_P2(self, set_test_backend, hubble):
        """Tests the second principal plane location calculation."""
        assert_allclose(operand.ParaxialOperand.P2(hubble), -57600.06413416607)

    def test_N1(self, set_test_backend, hubble):
        """Tests the first nodal point location calculation."""
        assert_allclose(operand.ParaxialOperand.N1(hubble), -414291.8559514676)

    def test_N2(self, set_test_backend, hubble):
        """Tests the second nodal point location calculation."""
        assert_allclose(operand.ParaxialOperand.N2(hubble), -57600.06413416545)

    def test_EPL(self, set_test_backend, hubble):
        """Tests the entrance pupil location calculation."""
        assert_allclose(operand.ParaxialOperand.EPL(hubble), 4910.01016)

    def test_EPD(self, set_test_backend, hubble):
        """Tests the entrance pupil diameter calculation."""
        assert_allclose(operand.ParaxialOperand.EPD(hubble), 2400)

    def test_XPL(self, set_test_backend, hubble):
        """Tests the exit pupil location calculation."""
        assert_allclose(operand.ParaxialOperand.XPL(hubble), -6958.363551642134)

    def test_XPD(self, set_test_backend, hubble):
        """Tests the exit pupil diameter calculation."""
        assert_allclose(operand.ParaxialOperand.XPD(hubble), 289.93210961932573)

    def test_magnification(self, set_test_backend, hubble):
        """Tests the paraxial magnification calculation."""
        assert_allclose(operand.ParaxialOperand.magnification(hubble), 0.0)

    def test_total_track(self, set_test_backend, hubble):
        """Tests the total track length calculation."""
        assert_allclose(operand.ParaxialOperand.total_track(hubble), 6365.20955)


class TestAberrationOperand:
    """
    Tests for operands related to Seidel and chromatic aberrations.
    """

    def test_seidel(self, set_test_backend, hubble):
        """Tests the calculation of a specific Seidel aberration coefficient."""
        assert_allclose(
            operand.AberrationOperand.seidels(hubble, 0),
            0.0014539022855417389,
        )

    def test_TSC(self, set_test_backend, hubble):
        """Tests Transverse Spherical Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TSC(hubble, 2), 0.0)

    def test_SC(self, set_test_backend, hubble):
        """Tests Spherical Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.SC(hubble, 2), 0.0)

    def test_CC(self, set_test_backend, hubble):
        """Tests Comatic Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.CC(hubble, 2), 0.0)

    def test_TCC(self, set_test_backend, hubble):
        """Tests Transverse Comatic Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TCC(hubble, 1), 0.0)

    def test_TAC(self, set_test_backend, hubble):
        """Tests Transverse Astigmatic Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TAC(hubble, 1), 0.0)

    def test_AC(self, set_test_backend, hubble):
        """Tests Astigmatic Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.AC(hubble, 1), 0.0)

    def test_TPC(self, set_test_backend, hubble):
        """Tests Transverse Petzval Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TPC(hubble, 1), 0.0)

    def test_PC(self, set_test_backend, hubble):
        """Tests Petzval Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.PC(hubble, 1), 0.0)

    def test_DC(self, set_test_backend, hubble):
        """Tests Distortion Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.DC(hubble, 1), 0.0)

    def test_TAchC(self, set_test_backend, hubble):
        """Tests Transverse Achromatic Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TAchC(hubble, 1), 0.0)

    def test_LchC(self, set_test_backend, hubble):
        """Tests Longitudinal Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.LchC(hubble, 1), 0.0)

    def test_TchC(self, set_test_backend, hubble):
        """Tests Transverse Chromatic aberration for a surface."""
        assert_allclose(operand.AberrationOperand.TchC(hubble, 1), 0.0)

    def test_TSC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Spherical Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TSC_sum(hubble), 0.0)

    def test_SC_sum(self, set_test_backend, hubble):
        """Tests the sum of Spherical Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.SC_sum(hubble), 0.0)

    def test_CC_sum(self, set_test_backend, hubble):
        """Tests the sum of Comatic Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.CC_sum(hubble), 0.0)

    def test_TCC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Comatic Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TCC_sum(hubble), 0.0)

    def test_TAC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Astigmatic Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TAC_sum(hubble), 0.0)

    def test_AC_sum(self, set_test_backend, hubble):
        """Tests the sum of Astigmatic Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.AC_sum(hubble), 0.0)

    def test_TPC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Petzval Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TPC_sum(hubble), 0.0)

    def test_PC_sum(self, set_test_backend, hubble):
        """Tests the sum of Petzval Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.PC_sum(hubble), 0.0)

    def test_DC_sum(self, set_test_backend, hubble):
        """Tests the sum of Distortion Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.DC_sum(hubble), 0.03489370392123652)

    def test_TAchC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Achromatic Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TAchC_sum(hubble), 0.0)

    def test_LchC_sum(self, set_test_backend, hubble):
        """Tests the sum of Longitudinal Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.LchC_sum(hubble), 0.0)

    def test_TchC_sum(self, set_test_backend, hubble):
        """Tests the sum of Transverse Chromatic aberration."""
        assert_allclose(operand.AberrationOperand.TchC_sum(hubble), 0.0)


class TestRayOperand:
    """
    Tests for operands based on real ray tracing, such as ray intercepts,
    direction cosines, spot size, and OPD.
    """

    def test_x_intercept(self, set_test_backend, hubble):
        """Tests the global x-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 1.0, "Hy": 0.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.x_intercept(**data), 150.42338010762842)

    def test_y_intercept(self, set_test_backend, hubble):
        """Tests the global y-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.y_intercept(**data), 150.42338010762842)

    def test_z_intercept(self, set_test_backend, hubble):
        """Tests the global z-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.z_intercept(**data), 6347.146837237045)

    def test_x_intercept_lcs(self, set_test_backend, hubble):
        """Tests the local x-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 1.0, "Hy": 0.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.x_intercept_lcs(**data), 150.42338010762842)

    def test_y_intercept_lcs(self, set_test_backend, hubble):
        """Tests the local y-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.y_intercept_lcs(**data), 150.42338010762842)

    def test_z_intercept_lcs(self, set_test_backend, hubble):
        """Tests the local z-coordinate intercept of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.z_intercept_lcs(**data), -18.062712762936826)

    def test_L(self, set_test_backend, hubble):
        """Tests the L direction cosine of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.L(**data), 0.0)

    def test_M(self, set_test_backend, hubble):
        """Tests the M direction cosine of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.M(**data), 0.021668820646914833)

    def test_N(self, set_test_backend, hubble):
        """Tests the N direction cosine of a ray."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "Px": 0.0, "Py": 0.0, "wavelength": 0.55}
        assert_allclose(RayOperand.N(**data), 0.9997652035411982)

    def test_rms_spot_size(self, set_test_backend, hubble):
        """Tests the RMS spot size for a single wavelength."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "wavelength": 0.55, "num_rays": 100}
        assert_allclose(RayOperand.rms_spot_size(**data), 0.025626727777956947)

    def test_rms_spot_size_all_wave(self, set_test_backend, hubble):
        """Tests the RMS spot size across all defined wavelengths."""
        data = {"optic": hubble, "surface_number": -1, "Hx": 0.0, "Hy": 1.0, "wavelength": "all", "num_rays": 100}
        assert_allclose(RayOperand.rms_spot_size(**data), 0.025626727777956947)

    def test_opd_diff(self, set_test_backend, hubble):
        """Tests the Optical Path Difference (OPD) for an off-axis field."""
        data = {"optic": hubble, "Hx": 0.0, "Hy": 1.0, "wavelength": 0.55, "num_rays": 3}
        assert_allclose(RayOperand.OPD_difference(**data), 0.0732334055041471)

    def test_opd_diff_on_axis(self, set_test_backend, hubble):
        """Tests the Optical Path Difference (OPD) for an on-axis field."""
        data = {"optic": hubble, "Hx": 0.0, "Hy": 0.0, "wavelength": 0.55, "num_rays": 3}
        assert_allclose(RayOperand.OPD_difference(**data), 0.0013295118803701674)

    def test_opd_diff_new_dist(self, set_test_backend, hubble):
        """Tests OPD with a different pupil sampling distribution."""
        data = {"optic": hubble, "Hx": 0.0, "Hy": 1.0, "wavelength": 0.55, "num_rays": 100, "distribution": "hexapolar"}
        assert_allclose(RayOperand.OPD_difference(**data), 0.2211995620762635)

    def create_test_optic(self):
        """Creates a Three-Mirror Anastigmat (TMA) for clearance testing."""
        lens = Optic(name="TMA")
        lens.set_aperture(aperture_type="EPD", value=10)
        lens.set_field_type(field_type="angle")
        lens.add_field(y=0); lens.add_field(y=+1.5); lens.add_field(y=-1.5)
        lens.add_wavelength(value=0.55, is_primary=True)
        lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
        lens.add_surface(index=1, radius=-100, thickness=-20, material="mirror", rx=be.radians(-15.0), is_stop=True)
        lens.add_surface(index=2, radius=-100, thickness=+20, material="mirror", rx=be.radians(-10.0), dy=-11.5)
        lens.add_surface(index=3, radius=-100, thickness=-19, material="mirror", rx=be.radians(-1.0), dy=-15)
        lens.add_surface(index=4, dy=-19.3)
        return lens

    def test_clearance(self, set_test_backend):
        """Tests the clearance operand between two rays."""
        optic = self.create_test_optic()
        dist1 = RayOperand.clearance(optic, 1, (0,0), (0,-1), 3, (0,1), (0,0), 0.55)
        assert_allclose(dist1, -7.412094834746042)
        dist2 = RayOperand.clearance(optic, 1, (0,0), (0,0), 3, (0,0), (0,0), 0.55)
        assert_allclose(dist2, -13.065596389231768)
        dist3 = RayOperand.clearance(optic, 1, (0,-1), (0,1), 3, (0,1), (0,0), 0.55)
        assert_allclose(dist3, -15.730530102711754)

    def test_AOI(self, set_test_backend, cooke_triplet):
        """Tests the Angle of Incidence (AOI) operand."""
        data1 = {"optic": cooke_triplet, "surface_number": 1, "Hx": 0, "Hy": 0.5, "Px": 0, "Py": 0, "wavelength": 0.55}
        aoi1 = RayOperand.AOI(**data1)
        data2 = {"optic": cooke_triplet, "surface_number": 7, "Hx": 0, "Hy": 1.0, "Px": 0, "Py": 0, "wavelength": 0.55}
        aoi2 = RayOperand.AOI(**data2)
        data3 = {"optic": cooke_triplet, "surface_number": 7, "Hx": 0, "Hy": 0.0, "Px": 0, "Py": 1.0, "wavelength": 0.55}
        aoi3 = RayOperand.AOI(**data3)
        assert_allclose(aoi1, 4.751694988282298)
        assert_allclose(aoi2, 18.98548831778711)
        assert_allclose(aoi3, 5.764367266821943)


class TestLensOperand:
    """
    Tests for operands related to the physical properties of lenses, such as
    edge thickness.
    """
    def setup_method(self):
        """Sets up a basic optic for lens operand tests."""
        self.optic = Optic()
        self.optic.set_aperture(aperture_type="EPD", value=20.0)
        self.optic.add_surface(index=0, thickness=be.inf)
        self.optic.add_surface(index=1, thickness=10.0, is_stop=True)
        self.optic.add_wavelength(value=0.550, is_primary=True)
        self.optic.set_field_type(field_type="angle")
        self.optic.add_field(y=0.0)

    def test_edge_thickness_simple_biconvex(self, set_test_backend):
        """Tests edge thickness for a simple biconvex lens."""
        self.optic.add_surface(index=2, radius=50.0, thickness=5.0, material="N-BK7")
        self.optic.add_surface(index=3, radius=-50.0, thickness=100.0)
        self.optic.add_surface(index=4)
        self.optic.update_paraxial()
        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)
        assert_allclose(edge_thickness, 2.97484176)

    def test_edge_thickness_plano_concave(self, set_test_backend):
        """Tests edge thickness for a plano-concave lens."""
        self.optic.add_surface(index=2, radius=be.inf, thickness=2.0, material="N-BK7")
        self.optic.add_surface(index=3, radius=50.0, thickness=100.0)
        self.optic.add_surface(index=4)
        self.optic.update_paraxial()
        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)
        assert_allclose(edge_thickness, 3.010075728)

    def test_edge_thickness_different_semi_apertures(self, set_test_backend):
        """Tests edge thickness when lens surfaces have different apertures."""
        self.optic.add_surface(index=2, radius=50.0, thickness=5.0, material="N-BK7")
        self.optic.add_surface(index=3, radius=-50.0, thickness=100.0)
        self.optic.add_surface(index=4)
        self.optic.update_paraxial()
        self.optic.surface_group.surfaces[2].semi_aperture = 10.0
        self.optic.surface_group.surfaces[3].semi_aperture = 8.0
        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)
        assert_allclose(edge_thickness, 3.6934855)

    def test_edge_thickness_aspheric_surface(self, set_test_backend):
        """Tests edge thickness for a lens with an aspheric surface."""
        self.optic.add_surface(index=2, surface_type="even_asphere", radius=50.0, thickness=5.0, material="N-BK7", coefficients=[1e-5])
        self.optic.add_surface(index=3, radius=-50.0, thickness=100.0)
        self.optic.add_surface(index=4)
        self.optic.update_paraxial()
        edge_thickness = LensOperand.edge_thickness(self.optic, surface_number=2)
        assert_allclose(edge_thickness, 2.97384176)


class TestOperand:
    """
    Tests the core Operand class, which wraps operand functions and handles
    targets, weights, and bounds.
    """

    def test_get_value(self, set_test_backend, hubble):
        """Tests that the `value` property correctly calls the operand."""
        op = operand.Operand(operand_type="f2", input_data={"optic": hubble})
        assert_allclose(op.value, 57600.080998403595)

    def test_invalid_operand(self, set_test_backend, hubble):
        """Tests that using an unregistered operand type raises a ValueError."""
        op = operand.Operand(operand_type="f3", input_data={"optic": hubble})
        with pytest.raises(ValueError):
            op.value

    def test_delta(self, set_test_backend, hubble):
        """Tests the delta calculation (value - target)."""
        op = operand.Operand(operand_type="f2", target=5000, input_data={"optic": hubble})
        assert_allclose(op.delta(), 52600.080998403595)

    def test_fun(self, set_test_backend, hubble):
        """Tests the weighted delta calculation (weight * delta)."""
        op = operand.Operand(operand_type="f2", target=1e5, weight=1.5, input_data={"optic": hubble})
        assert_allclose(op.fun(), -63599.87850239461)

    def test_reregister_operand(self, set_test_backend):
        """Tests that attempting to re-register an existing operand raises an error."""
        try:
            operand.operand_registry._registry.pop("f3")
        except KeyError:
            pass
        operand.operand_registry.register("f3", lambda x: x)
        with pytest.raises(ValueError):
            operand.operand_registry.register("f3", lambda x: x)

    def test_contains(self, set_test_backend):
        """Tests the `in` operator for the operand registry."""
        assert "f1" in operand.operand_registry
        assert "invalid" not in operand.operand_registry

    def test_repr(self, set_test_backend):
        """Tests the string representation of the operand registry."""
        assert isinstance(repr(operand.operand_registry), str)

    def test_invalid_bounds(self, set_test_backend):
        """Tests that setting min_val > max_val raises a ValueError."""
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0)

    def test_invalid_bounds_and_target(self, set_test_backend):
        """Tests that providing both bounds and a target raises a ValueError."""
        with pytest.raises(ValueError):
            operand.Operand(operand_type="f1", min_val=1, max_val=0, target=1)

    def test_delta_ineq(self, set_test_backend):
        """Tests the inequality delta calculation for bounded operands."""
        input_data = {"optic": HubbleTelescope()}
        op_lower = operand.Operand(operand_type="f2", min_val=1e6, max_val=1e11, input_data=input_data)
        op_upper = operand.Operand(operand_type="f2", min_val=0, max_val=1, input_data=input_data)
        f2 = HubbleTelescope().paraxial.f2()
        assert_allclose(op_lower.delta_ineq(), 1e6 - f2)
        assert_allclose(op_upper.delta_ineq(), f2 - 1)

    def test_delta_ineq_no_bounds(self, set_test_backend):
        """Tests that inequality delta is zero when no bounds are set."""
        op = operand.Operand(operand_type="f2", input_data={"optic": HubbleTelescope()})
        assert_allclose(op.delta_ineq(), 0)

    def test_delta_with_bounds(self, set_test_backend):
        """Tests that the `delta` method considers bounds if no target is set."""
        input_data = {"optic": HubbleTelescope()}
        op = operand.Operand(operand_type="f2", min_val=1e6, max_val=1e11, input_data=input_data)
        f2 = HubbleTelescope().paraxial.f2()
        assert_allclose(op.delta(), 1e6 - f2)

    def test_invalid_delta(self, set_test_backend):
        """Tests that calling `delta` without a target or bounds raises an error."""
        op = operand.Operand(operand_type="f2", input_data={"optic": HubbleTelescope()})
        op.target = None
        with pytest.raises(ValueError):
            op.delta()


class TestOperandManager:
    """
    Tests the OperandManager class, which manages a collection of operands
    for an optimization task.
    """

    def test_add(self, set_test_backend):
        """Tests adding an operand to the manager."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert len(manager) == 1

    def test_clear(self, set_test_backend):
        """Tests clearing all operands from the manager."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager.clear()
        assert len(manager) == 0

    def test_iter(self, set_test_backend):
        """Tests iterating through the operands in the manager."""
        manager = operand.OperandManager()
        manager.add("f1", 1); manager.add("f2", 1)
        for op in manager:
            assert isinstance(op, operand.Operand)

    def test_getitem(self, set_test_backend):
        """Tests accessing an operand by index."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        assert isinstance(manager[0], operand.Operand)

    def test_setitem(self, set_test_backend):
        """Tests replacing an operand at a specific index."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        manager[0] = operand.Operand(operand_type="f2", target=1)
        assert manager[0].operand_type == "f2"

    def test_len(self, set_test_backend):
        """Tests the length of the manager."""
        manager = operand.OperandManager()
        manager.add("f1", 1); manager.add("f2", 1)
        assert len(manager) == 2

    def test_getitem_error(self, set_test_backend):
        """Tests that accessing an out-of-bounds index raises an error."""
        manager = operand.OperandManager()
        with pytest.raises(IndexError):
            manager[0]

    def test_setitem_invalid_type(self, set_test_backend):
        """Tests that setting an item to a non-Operand type raises an error."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        with pytest.raises(ValueError):
            manager[0] = 1

    def test_delitem(self, set_test_backend):
        """Tests deleting an operand from the manager."""
        manager = operand.OperandManager()
        manager.add("f1", 1)
        del manager[0]
        assert len(manager) == 0