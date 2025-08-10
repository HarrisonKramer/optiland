import pytest
import optiland.backend as be
from optiland.wavefront.reference_sphere import (
    ChiefRayReferenceSphereCalculator,
    create_reference_sphere_calculator,
)
from optiland.samples.objectives import DoubleGauss

from tests.utils import assert_allclose


class TestChiefRayReferenceSphereCalculator:
    def test_get_reference_sphere(self, set_test_backend):
        optic = DoubleGauss()
        optic.trace_generic(0, 0, Px=0.0, Py=0.0, wavelength=0.55)
        calculator = ChiefRayReferenceSphereCalculator(optic)
        xc, yc, zc, R = calculator.calculate()
        assert_allclose(xc, 0)
        assert_allclose(yc, 0)
        assert_allclose(zc, 139.454938)
        assert_allclose(R, 114.64441695)

    def test_get_reference_sphere_error(self, set_test_backend):
        optic = DoubleGauss()
        optic.trace(Hx=0, Hy=0, wavelength=0.55)
        calculator = ChiefRayReferenceSphereCalculator(optic)
        # fails when >1 rays traced in the pupil
        with pytest.raises(ValueError):
            calculator.calculate()


class TestCreateReferenceSphereCalculator:
    def test_create_chief_ray_calculator(self, set_test_backend):
        optic = DoubleGauss()
        calculator = create_reference_sphere_calculator("chief_ray", optic)
        assert isinstance(calculator, ChiefRayReferenceSphereCalculator)

    def test_create_unknown_calculator(self, set_test_backend):
        optic = DoubleGauss()
        with pytest.raises(ValueError):
            create_reference_sphere_calculator("unknown", optic)
