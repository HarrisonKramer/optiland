import pytest
import optiland.backend as be
from optiland.wavefront.reference_sphere import (
    ChiefRayReferenceSphereCalculator,
    create_reference_sphere_calculator,
)
from optiland.samples.objectives import DoubleGauss


class TestChiefRayReferenceSphereCalculator:
    def test_get_reference_sphere(self, set_test_backend):
        optic = DoubleGauss()
        optic.trace_generic(0, 0, Px=0.0, Py=0.0, wavelength=0.55)
        calculator = ChiefRayReferenceSphereCalculator(optic)
        xc, yc, zc, R = calculator.calculate(pupil_z=100)
        assert be.allclose(xc, be.array([0.0]))
        assert be.allclose(yc, be.array([0.0]))
        assert be.allclose(zc, be.array([139.454938]))
        assert be.allclose(R, be.array([39.454938]))

    def test_get_reference_sphere_error(self, set_test_backend):
        optic = DoubleGauss()
        optic.trace(Hx=0, Hy=0, wavelength=0.55)
        calculator = ChiefRayReferenceSphereCalculator(optic)
        # fails when >1 rays traced in the pupil
        with pytest.raises(ValueError):
            calculator.calculate(pupil_z=100)


class TestCreateReferenceSphereCalculator:
    def test_create_chief_ray_calculator(self, set_test_backend):
        optic = DoubleGauss()
        calculator = create_reference_sphere_calculator("chief_ray", optic)
        assert isinstance(calculator, ChiefRayReferenceSphereCalculator)

    def test_create_unknown_calculator(self, set_test_backend):
        optic = DoubleGauss()
        with pytest.raises(ValueError):
            create_reference_sphere_calculator("unknown", optic)
