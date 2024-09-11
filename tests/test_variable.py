import pytest
import numpy as np
from optiland import variable
from optiland.geometries import PolynomialGeometry
from optiland.coordinate_system import CoordinateSystem
from optiland.samples.microscopes import Objective60x, UVReflectingMicroscope
from optiland.samples.simple import AsphericSinglet


class TestRadiusVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.radius_var = variable.RadiusVariable(self.optic, 1)

    def test_get_value(self):
        assert np.isclose(self.radius_var.get_value(), 4.5325999999999995)

    def test_update_value(self):
        self.radius_var.update_value(5.0)
        assert np.isclose(self.radius_var.get_value(), 5.0)


class TestConicVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = UVReflectingMicroscope()
        self.conic_var = variable.ConicVariable(self.optic, 1)

    def test_get_value(self):
        assert np.isclose(self.conic_var.get_value(), 0.0)

    def test_update_value(self):
        self.conic_var.update_value(-0.5)
        assert np.isclose(self.conic_var.get_value(), -0.5)


class TestThicknessVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.thickness_var = variable.ThicknessVariable(self.optic, 2)

    def test_get_value(self):
        assert np.isclose(self.thickness_var.get_value(), -0.5599999999999994)

    def test_update_value(self):
        self.thickness_var.update_value(-0.6)
        assert np.isclose(self.thickness_var.get_value(), -0.6)


class TestIndexVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = Objective60x()
        self.index_var = variable.IndexVariable(self.optic, 1, 0.55)

    def test_get_value(self):
        assert np.isclose(self.index_var.get_value(), -0.012206444700957775)

    def test_update_value(self):
        self.index_var.update_value(0.1)
        assert np.isclose(self.index_var.get_value(), 0.1)


class TestAsphereCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        self.asphere_var = variable.AsphereCoeffVariable(self.optic, 1, 0)

    def test_get_value(self):
        assert np.isclose(self.asphere_var.get_value(), -2.248851)

    def test_update_value(self):
        self.asphere_var.update_value(-2.0)
        assert np.isclose(self.asphere_var.get_value(), -2.0)


class TestPolynomialCoeffVariable:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.optic = AsphericSinglet()
        poly_geo = PolynomialGeometry(CoordinateSystem(), 100,
                                      coefficients=np.zeros((3, 3)))
        self.optic.surface_group.surfaces[0].geometry = poly_geo
        self.poly_var = variable.PolynomialCoeffVariable(self.optic, 0, (1, 1))

    def test_get_value(self):
        assert self.poly_var.get_value() == 0.0

    def test_update_value(self):
        self.poly_var.update_value(1.0)
        assert np.isclose(self.poly_var.get_value(), 1.0)
