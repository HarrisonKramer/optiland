# tests/optimization/test_variable.py
"""
Tests for the variable classes in optiland.optimization.variable.

These classes wrap the parameters of an optical system (e.g., radius,
thickness, material) to make them accessible to optimization algorithms.
"""
import optiland.backend as be
import pytest
from unittest.mock import patch

from optiland.coordinate_system import CoordinateSystem
from optiland.geometries import (
    ChebyshevPolynomialGeometry,
    ForbesQbfsGeometry,
    ForbesQ2dGeometry,
    PolynomialGeometry,
    ZernikePolynomialGeometry,
    ForbesSurfaceConfig,
)
from optiland.optimization import variable, OptimizationProblem, OptimizerGeneric
from optiland.samples.microscopes import Objective60x, UVReflectingMicroscope
from optiland.samples.simple import AsphericSinglet, Edmund_49_847
from optiland.materials.abbe import AbbeMaterial
from optiland.optimization.variable.material import MaterialVariable
from optiland.optimization.scaling.identity import IdentityScaler
from ..utils import assert_allclose


class TestRadiusVariable:
    """Tests the RadiusVariable for standard radius of curvature."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.radius_var = variable.RadiusVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the radius value."""
        assert_allclose(self.radius_var.get_value(), 553.260)

    def test_update_value(self, set_test_backend):
        """Tests updating the radius value."""
        self.radius_var.update_value(5.0)
        assert_allclose(self.radius_var.get_value(), 5.0)


class TestReciprocalRadiusVariable:
    """
    Tests the ReciprocalRadiusVariable, which uses curvature (1/R) as the
    optimization parameter.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Edmund_49_847()
        self.reciprocal_radius_var = variable.ReciprocalRadiusVariable(self.optic, 1)
        self.radius_var = variable.RadiusVariable(self.optic, 1, scaler=IdentityScaler())
        self.problem = OptimizationProblem(optic=self.optic)
        self.problem.add_operand("rms_spot_size", target=0, weight=1, Hx=0, Hy=0, num_rays=5, distribution="hexapolar", surface_number=3)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the curvature value."""
        expected = 1.0 / self.radius_var.get_value()
        assert_allclose(self.reciprocal_radius_var.get_value(), expected, atol=1e-4)

    def test_update_value(self, set_test_backend):
        """Tests updating the radius via its curvature."""
        self.reciprocal_radius_var.update_value(0.008)
        assert_allclose(self.radius_var.get_value(), 1.0 / 0.008, atol=1e-4)

    def test_get_value_infinity(self, set_test_backend):
        """Tests that a radius of zero corresponds to infinite curvature."""
        self.radius_var.update_value(0.0)
        assert self.reciprocal_radius_var.get_value() == be.inf

    def test_update_value_zero(self, set_test_backend):
        """Tests that a curvature of zero corresponds to infinite radius."""
        self.reciprocal_radius_var.update_value(0.0)
        assert self.radius_var.get_value() == be.inf

    def test_optimization(self):
        """Tests optimizing the reciprocal radius."""
        self.problem.add_variable("reciprocal_radius", surface_number=1)
        optimizer = OptimizerGeneric(self.problem)
        self.optic.set_radius(22.0, 1)
        optimizer.optimize(tol=1e-9)
        assert_allclose(self.optic.surface_group.radii[1], 19.93, atol=5)

    def test_optimization_with_flat_surface(self):
        """Tests optimizing starting from a flat surface (inf radius)."""
        self.problem.add_variable("reciprocal_radius", surface_number=1)
        self.optic.set_radius(-be.inf, 1)
        optimizer = OptimizerGeneric(self.problem)
        optimizer.optimize(tol=1e-9)
        assert_allclose(self.optic.surface_group.radii[1], 19.93, atol=5)


class TestConicVariable:
    """Tests the ConicVariable for the conic constant."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = UVReflectingMicroscope()
        self.conic_var = variable.ConicVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the conic constant value."""
        assert_allclose(self.conic_var.get_value(), 0.0)

    def test_update_value(self, set_test_backend):
        """Tests updating the conic constant value."""
        self.conic_var.update_value(-0.5)
        assert_allclose(self.conic_var.get_value(), -0.5)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.conic_var) == "Conic Constant, Surface 1"


class TestThicknessVariable:
    """Tests the ThicknessVariable."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the thickness value."""
        assert_allclose(self.thickness_var.get_value(), 4.4)

    def test_update_value(self, set_test_backend):
        """Tests updating the thickness value."""
        self.thickness_var.update_value(5.0)
        assert_allclose(self.thickness_var.get_value(), 5.0)


class TestIndexVariable:
    """Tests the IndexVariable for refractive index."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the refractive index value."""
        assert_allclose(self.index_var.get_value(), 1.4877935552990422)

    def test_update_value(self, set_test_backend):
        """Tests updating the refractive index value."""
        self.index_var.update_value(1.6)
        assert_allclose(self.index_var.get_value(), 1.6)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.index_var) == "Refractive Index, Surface 1"


class TestMaterialVariable:
    """Tests the MaterialVariable for glass selection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.glass_selection = ["N-BK7", "N-SSK2", "N-SK2", "N-SK16"]
        self.material_var = variable.MaterialVariable(self.optic, 1, self.glass_selection)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the material name."""
        assert self.material_var.get_value() == "N-FK51"

    def test_update_value(self, set_test_backend):
        """Tests updating the material."""
        self.material_var.update_value("F5")
        assert self.material_var.get_value() == "F5"

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.material_var) == "Material, Surface 1"

    def test_init_with_abbe_material(self):
        """
        Tests initializing a MaterialVariable on a surface that has an
        AbbeMaterial, ensuring it finds the closest real glass.
        """
        self.optic.surface_group.surfaces[1].material_post = AbbeMaterial(n=1.5168, abbe=64.17)
        with patch("optiland.materials.material_utils.find_closest_glass", return_value="N-BK7") as mock_find:
            mat_var = MaterialVariable(self.optic, 1, self.glass_selection)
            assert mat_var.get_value() == "N-BK7"
            mock_find.assert_called()


class TestAsphereCoeffVariable:
    """Tests the AsphereCoeffVariable for aspheric coefficients."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0)

    def test_get_value(self, set_test_backend):
        """Tests retrieving an aspheric coefficient value."""
        assert_allclose(self.asphere_var.get_value(), -0.0002248851)

    def test_update_value(self, set_test_backend):
        """Tests updating an aspheric coefficient value."""
        self.asphere_var.update_value(-2.0)
        assert_allclose(self.asphere_var.get_value(), -2.0)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.asphere_var) == "Asphere Coeff. 0, Surface 1"


class TestPolynomialCoeffVariable:
    """Tests the PolynomialCoeffVariable for XY polynomial surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100, coefficients=be.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self, set_test_backend):
        """Tests retrieving a polynomial coefficient value."""
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        """Tests updating a polynomial coefficient value."""
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.poly_var) == "Poly. Coeff. (1, 1), Surface 0"


class TestChebyshevCoeffVariable:
    """Tests the ChebyshevCoeffVariable for Chebyshev polynomial surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ChebyshevPolynomialGeometry(CoordinateSystem(), 100, coefficients=be.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ChebyshevCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self, set_test_backend):
        """Tests retrieving a Chebyshev coefficient value."""
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        """Tests updating a Chebyshev coefficient value."""
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.poly_var) == "Chebyshev Coeff. (1, 1), Surface 0"


class TestZernikeCoeffVariable:
    """Tests the ZernikeCoeffVariable for Zernike polynomial surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100, coefficients=be.zeros(3))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.ZernikeCoeffVariable(self.optic, 0, 1)

    def test_get_value(self, set_test_backend):
        """Tests retrieving a Zernike coefficient value."""
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self, set_test_backend):
        """Tests updating a Zernike coefficient value."""
        self.poly_var.update_value(1.0)
        assert_allclose(self.poly_var.get_value(), 1.0)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.poly_var) == "Zernike Coeff. 1, Surface 0"


class TestVariable:
    """Tests the generic Variable factory class."""

    def test_get_value(self, set_test_backend):
        """Tests creating a variable and getting its value."""
        optic = Objective60x()
        radius_var = variable.Variable(optic, "radius", surface_number=1)
        assert_allclose(radius_var.value, 4.5326)

    def test_unrecognized_attribute(self, capsys):
        """Tests that an unrecognized attribute keyword raises a warning."""
        optic = Objective60x()
        variable.Variable(optic, "radius", 1, unrecognized_attribute=1)
        captured = capsys.readouterr()
        assert "Warning: unrecognized_attribute is not a recognized attribute" in captured.out

    def test_invalid_type(self, set_test_backend):
        """Tests that creating a variable with an invalid type raises an error."""
        with pytest.raises(ValueError):
            variable.Variable(Objective60x(), "invalid", surface_number=1)


class TestTiltVariable:
    """Tests the TiltVariable for surface tilts."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.tilt_var_x = variable.TiltVariable(self.optic, 1, "x")
        self.tilt_var_y = variable.TiltVariable(self.optic, 1, "y")

    def test_get_value(self, set_test_backend):
        """Tests retrieving the tilt value for both axes."""
        assert_allclose(self.tilt_var_x.get_value(), 0.0)
        assert_allclose(self.tilt_var_y.get_value(), 0.0)

    def test_update_value(self, set_test_backend):
        """Tests updating the tilt value for both axes."""
        self.tilt_var_x.update_value(5.0)
        assert_allclose(self.tilt_var_x.get_value(), 5.0)
        self.tilt_var_y.update_value(-5.0)
        assert_allclose(self.tilt_var_y.get_value(), -5.0)

    def test_invalid_axis(self, set_test_backend):
        """Tests that creating a tilt variable with an invalid axis raises an error."""
        with pytest.raises(ValueError):
            variable.TiltVariable(self.optic, 1, "z")

    def test_str(self, set_test_backend):
        """Tests the string representation of the tilt variables."""
        assert str(self.tilt_var_x) == "Tilt X, Surface 1"
        assert str(self.tilt_var_y) == "Tilt Y, Surface 1"


class TestDecenterVariable:
    """Tests the DecenterVariable for surface decenters."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.decenter_var_x = variable.DecenterVariable(self.optic, 1, "x")
        self.decenter_var_y = variable.DecenterVariable(self.optic, 1, "y")

    def test_get_value(self, set_test_backend):
        """Tests retrieving the decenter value for both axes."""
        assert_allclose(self.decenter_var_x.get_value(), 0.0)
        assert_allclose(self.decenter_var_y.get_value(), 0.0)

    def test_update_value(self, set_test_backend):
        """Tests updating the decenter value for both axes."""
        self.decenter_var_x.update_value(5.0)
        assert_allclose(self.decenter_var_x.get_value(), 5.0)
        self.decenter_var_y.update_value(-5.0)
        assert_allclose(self.decenter_var_y.get_value(), -5.0)

    def test_invalid_axis(self, set_test_backend):
        """Tests that creating a decenter variable with an invalid axis raises an error."""
        with pytest.raises(ValueError):
            variable.DecenterVariable(self.optic, 1, "z")

    def test_str(self, set_test_backend):
        """Tests the string representation of the decenter variables."""
        assert str(self.decenter_var_x) == "Decenter X, Surface 1"
        assert str(self.decenter_var_y) == "Decenter Y, Surface 1"


class TestNormalizationRadiusVariable:
    """Tests the NormalizationRadiusVariable for aspheric/freeform surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        zernike_geo = ZernikePolynomialGeometry(CoordinateSystem(), 100, norm_radius=10.0)
        self.optic.surface_group.surfaces[1].geometry = zernike_geo
        self.norm_radius_var = variable.NormalizationRadiusVariable(self.optic, 1)

    def test_get_value(self, set_test_backend):
        """Tests retrieving the normalization radius value."""
        assert_allclose(self.norm_radius_var.get_value(), 10.0)

    def test_update_value(self, set_test_backend):
        """Tests updating the normalization radius value."""
        self.norm_radius_var.update_value(12.0)
        assert_allclose(self.norm_radius_var.get_value(), 12.0)

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.norm_radius_var) == "Normalization Radius, Surface 1"


class TestForbesQbfsCoeffVariable:
    """Tests the ForbesQbfsCoeffVariable for Forbes Q-bfs surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        config = ForbesSurfaceConfig(radius=100, terms={1: 0.1, 2: 0.2, 3: 0.3}, norm_radius=15.0)
        forbes_geo = ForbesQbfsGeometry(CoordinateSystem(), surface_config=config)
        self.optic.surface_group.surfaces[1].geometry = forbes_geo
        self.forbes_var = variable.ForbesQbfsCoeffVariable(self.optic, 1, 1)

    def test_get_value(self, set_test_backend):
        """Tests retrieving a Forbes Q-bfs coefficient value."""
        assert_allclose(self.forbes_var.get_value(), 0.1)

    def test_update_value(self, set_test_backend):
        """Tests updating an existing Forbes Q-bfs coefficient."""
        self.forbes_var.update_value(0.5)
        assert_allclose(self.forbes_var.get_value(), 0.5)

    def test_update_value_out_of_bounds(self, set_test_backend):
        """Tests updating a coefficient that extends the existing list."""
        forbes_var_new = variable.ForbesQbfsCoeffVariable(self.optic, 1, 4)
        forbes_var_new.update_value(0.9)
        assert_allclose(forbes_var_new.get_value(), 0.9)
        assert len(self.optic.surface_group.surfaces[1].geometry.coeffs_c) == 5

    def test_string_representation(self, set_test_backend):
        """Tests the string representation of the variable."""
        assert str(self.forbes_var) == "Forbes Q-bfs Coeff n=1, Surface 1"


class TestForbesQ2dCoeffVariable:
    """Tests the ForbesQ2dCoeffVariable for Forbes Q-2d surfaces."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        coeffs = {('a', 1, 1): 0.1, ('a', 2, 2): 0.2, ('b', 1, 1): 0.3}
        config = ForbesSurfaceConfig(radius=100, terms=coeffs, norm_radius=15.0)
        forbes_geo = ForbesQ2dGeometry(CoordinateSystem(), surface_config=config)
        self.optic.surface_group.surfaces[1].geometry = forbes_geo
        self.forbes_var = variable.ForbesQ2dCoeffVariable(self.optic, 1, ('a', 2, 2))

    def test_get_value(self, set_test_backend):
        """Tests retrieving a Forbes Q-2d coefficient."""
        assert_allclose(self.forbes_var.get_value(), 0.2)

    def test_update_value_new(self, set_test_backend):
        """Tests creating and updating a new Forbes Q-2d coefficient."""
        key = ('a', 1, 3)
        forbes_var_new = variable.ForbesQ2dCoeffVariable(self.optic, 1, key)
        forbes_var_new.update_value(0.9)
        assert_allclose(forbes_var_new.get_value(), 0.9)


class TestVariableManager:
    """Tests the VariableManager class for managing a list of variables."""

    def test_add(self, set_test_backend):
        """Tests adding a variable to the manager."""
        var_manager = variable.VariableManager(optic=Objective60x())
        var_manager.add("radius", surface_number=1)
        assert len(var_manager) == 1

    def test_clear(self, set_test_backend):
        """Tests clearing all variables from the manager."""
        var_manager = variable.VariableManager(optic=Objective60x())
        var_manager.add("radius", surface_number=1)
        var_manager.clear()
        assert len(var_manager) == 0

    def test_iter(self, set_test_backend):
        """Tests iterating over the variables in the manager."""
        var_manager = variable.VariableManager(optic=Objective60x())
        var_manager.add("radius", surface_number=1)
        var_manager.add("radius", surface_number=2)
        for var in var_manager:
            assert isinstance(var, variable.Variable)

    def test_setitem(self, set_test_backend):
        """Tests replacing a variable at a specific index."""
        optic = Objective60x()
        var_manager = variable.VariableManager(optic=optic)
        var_manager.add("radius", surface_number=1)
        var_manager[0] = variable.Variable(optic, "radius", surface_number=2)
        assert var_manager[0].variable.surface_number == 2

    def test_delitem(self, set_test_backend):
        """Tests deleting a variable from the manager."""
        var_manager = variable.VariableManager(optic=Objective60x())
        var_manager.add("radius", surface_number=1)
        del var_manager[0]
        assert len(var_manager) == 0