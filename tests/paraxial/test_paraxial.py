# tests/paraxial/test_paraxial.py
"""
Tests for the Paraxial class in optiland.paraxial.

This file contains a data-driven test suite that verifies the paraxial
calculations for a wide variety of sample optical systems against known,
correct values. This ensures the robustness of the paraxial engine.
"""
import pytest

import optiland.backend as be
from optiland.optic import Optic
from optiland.paraxial import Paraxial
from optiland.samples.eyepieces import EyepieceErfle
from optiland.samples.infrared import InfraredTriplet, InfraredTripletF4
from optiland.samples.miscellaneous import NavarroWideAngleEye
from optiland.samples.objectives import (
    CookeTriplet,
    DoubleGauss,
    HeliarLens,
    LensWithFieldCorrector,
    ObjectiveUS008879901,
    PetzvalLens,
    ReverseTelephoto,
    Telephoto,
    TelescopeObjective48Inch,
    TessarLens,
    TripletTelescopeObjective,
)
from optiland.samples.simple import (
    CementedAchromat,
    Edmund_49_847,
    SingletStopSurf2,
    TelescopeDoublet,
)
from optiland.samples.telescopes import HubbleTelescope
from ..utils import assert_allclose


def get_optic_data():
    """
    Returns a list of tuples, each containing an optical system class and a
    dictionary of its expected paraxial properties. This data is used to
    parametrize the tests.
    """
    # Data format: (OpticClass, {property_name: expected_value, ...})
    return [
        (EyepieceErfle, {"f1": -79.687, "f2": 79.687, "F1": -18.953, "F2": 0.412, "P1": 60.734, "P2": -79.275, "EPL": 0.0, "EPD": 4.0, "XPL": -334.624, "XPD": 16.817, "FNO": 19.921, "invariant": -0.727}),
        (HubbleTelescope, {"f1": -57600.08, "f2": 57600.08, "F1": -471891.9, "F2": 0.0168, "P1": -414291.8, "P2": -57600.06, "EPL": 4910.01, "EPD": 2400, "XPL": -6958.36, "XPD": 289.93, "FNO": 24.0, "invariant": -3.141}),
        (InfraredTriplet, {"f1": -10.002, "f2": 10.002, "F1": -1.528, "F2": 0.0079, "P1": 8.474, "P2": -9.994, "EPL": 0.0, "EPD": 5.001, "XPL": -65.468, "XPD": 32.738, "FNO": 2.0, "invariant": -0.174}),
        (CookeTriplet, {"f1": -49.999, "f2": 49.999, "F1": -37.345, "F2": 0.207, "P1": 12.654, "P2": -49.792, "EPL": 11.512, "EPD": 10.0, "XPL": -50.961, "XPD": 10.233, "FNO": 4.999, "invariant": -1.819}),
    ]


@pytest.fixture
def optic_and_values(set_test_backend, request):
    """
    A fixture that receives a tuple from the parametrized data (optic class
    and expected values) and returns an instance of the optic and the values.
    """
    cls, values = request.param
    return cls(), values


def test_paraxial_init(set_test_backend):
    """Tests the initialization of the Paraxial class."""
    optic = Optic()
    paraxial = Paraxial(optic)
    assert paraxial.optic == optic
    assert paraxial.surfaces == optic.surface_group


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_f1(optic_and_values):
    """Tests the front focal length calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.f1(), values["f1"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_f2(optic_and_values):
    """Tests the back focal length calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.f2(), values["f2"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_F1(optic_and_values):
    """Tests the front focal point calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.F1(), values["F1"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_F2(optic_and_values):
    """Tests the back focal point calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.F2(), values["F2"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_P1(optic_and_values):
    """Tests the first principal plane calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.P1(), values["P1"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_P2(optic_and_values):
    """Tests the second principal plane calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.P2(), values["P2"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_EPL(optic_and_values):
    """Tests the entrance pupil location calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.EPL(), values["EPL"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_EPD(optic_and_values):
    """Tests the entrance pupil diameter calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.EPD(), values["EPD"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_XPL(optic_and_values):
    """Tests the exit pupil location calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.XPL(), values["XPL"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_XPD(optic_and_values):
    """Tests the exit pupil diameter calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.XPD(), values["XPD"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_FNO(optic_and_values):
    """Tests the F-number calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.FNO(), values["FNO"], atol=1e-2)


@pytest.mark.parametrize("optic_and_values", get_optic_data(), indirect=True)
def test_calculate_invariant(optic_and_values):
    """Tests the Lagrange invariant calculation for multiple systems."""
    optic_instance, values = optic_and_values
    assert_allclose(optic_instance.paraxial.invariant(), values["invariant"], atol=1e-2)


def test_EPD_float_by_stop_size(set_test_backend):
    """
    Tests the EPD calculation when the aperture type is 'float_by_stop_size'.
    """
    lens = CookeTriplet()
    lens.set_aperture(aperture_type="float_by_stop_size", value=7.6)
    assert_allclose(lens.paraxial.EPD(), 9.997764563903155)