import pytest
import numpy as np
from optiland.samples.simple import Edmund_49_847, SingletStopSurf2
from optiland.samples.objectives import (
    Cooke_triplet,
    double_Gauss,
    reverse_telephoto
)
from optiland.samples.eyepieces import eyepiece_Erfle


@pytest.mark.parametrize('optic, target',
                         [(Edmund_49_847, ),
                          (SingletStopSurf2, ),
                          (Cooke_triplet, ),
                          (double_Gauss, ),
                          (reverse_telephoto, ),
                          (eyepiece_Erfle, )])
def test_f1(optic, target):
    assert np.allclose(optic.paraxial.f1(), target)


def test_f2(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_F1(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_F2(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_P1(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_P2(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_N1(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_N2(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_EPD(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_EPL(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_XPD(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_XPL(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_FNO(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)


def test_invariant(optic, target):
    assert np.allclose(optic.paraxial.f2(), target)



# class TestSingletObjInfinityStop1(unittest.TestCase):
#     def setUp(self):
#         air = materials.IdealMaterial(n=1, k=0)
#         mat = materials.IdealMaterial(n=1.5185223876, k=0)

#         self.lens = optic.Optic()

#         self.lens.surface_group.surfaces = []

#         self.lens.set_aperture(aperture_type='EPD', value=10)
#         self.lens.set_field_type(field_type='angle')

#         self.lens.add_field(y=0)
#         self.lens.add_field(y=7)
#         self.lens.add_field(y=10)

#         self.lens.add_surface(index=0, thickness=np.inf, radius=np.inf, conic=0, material=air)
#         self.lens.add_surface(index=1, thickness=10, radius=150, conic=0, material=mat, is_stop=True)
#         self.lens.add_surface(index=2, thickness=140, radius=-250, conic=0, material=air)
#         self.lens.add_surface(index=3)

#         self.lens.add_wavelength(0.55, is_primary=True)

#     def test_f1(self):
#         self.assertAlmostEquals(self.lens.paraxial.f1(), -182.358953, places=5)

#     def test_f2(self):
#         self.assertAlmostEquals(self.lens.paraxial.f2(), 182.358953, places=5)

#     def test_EPD(self):
#         self.assertAlmostEquals(self.lens.paraxial.EPD(), 10.0, places=5)

#     def test_EPL(self):
#         self.assertAlmostEquals(self.lens.paraxial.EPL(), 0.0, places=5)

#     def test_XPD(self):
#         self.assertAlmostEquals(self.lens.paraxial.XPD(), 10.13848, places=5)

#     def test_XPL(self):
#         self.assertAlmostEquals(self.lens.paraxial.XPL(), -146.6765413, places=5)

#     def test_magnification(self):
#         self.assertAlmostEqual(self.lens.paraxial.magnification(), 0, places=5)

#     def test_F1(self):
#         self.assertAlmostEqual(self.lens.paraxial.F1(), -179.868184, places=5)

#     def test_F2(self):
#         self.assertAlmostEquals(self.lens.paraxial.F2(), 38.207672, places=5)

#     def test_P1(self):
#         self.assertAlmostEquals(self.lens.paraxial.P1(), 2.490769, places=5)

#     def test_P2(self):
#         self.assertAlmostEquals(self.lens.paraxial.P2(), -144.151281, places=5)

#     def test_N1(self):
#         self.assertAlmostEquals(self.lens.paraxial.N1(), 2.490769, places=5)

#     def test_N2(self):
#         self.assertAlmostEquals(self.lens.paraxial.N2(), -144.151281, places=5)

#     def test_invariant(self):
#         self.assertAlmostEqual(self.lens.paraxial.invariant(), -0.8816349035423249, places=5)

#     def test_marginal_ray(self):
#         y, u = self.lens.paraxial.marginal_ray()
#         y_target = np.array([[5.], [5.], [4.8861783025], [1.0475951700]])
#         self.assertTrue(np.max(np.abs(y - y_target)) < 0.00001)
#         u_target = np.array([[0.], [-0.01138217], [-0.02741845], [-0.02741845]])
#         self.assertTrue(np.max(np.abs(u - u_target)) < 0.00001)

#     def test_chief_ray(self):
#         y, u = self.lens.paraxial.chief_ray()
#         y_target = np.array([[0.0], [0.0], [1.1611747192], [25.509778769]])
#         self.assertTrue(np.max(np.abs(y - y_target)) < 0.00001)
#         u_target = np.array([[0.17632698], [0.11611747], [0.1739186], [0.1739186]])
#         self.assertTrue(np.max(np.abs(u - u_target)) < 0.00001)


# class TestSingletObjInfinityStop2(unittest.TestCase):
#     def setUp(self):
#         air = materials.IdealMaterial(n=1, k=0)
#         mat = materials.IdealMaterial(n=1.5185223876, k=0)

#         self.lens = optic.Optic()

#         self.lens.surface_group.surfaces = []

#         self.lens.set_aperture(aperture_type='EPD', value=10)
#         self.lens.set_field_type(field_type='angle')

#         self.lens.add_field(y=0)
#         self.lens.add_field(y=7)
#         self.lens.add_field(y=10)

#         self.lens.add_surface(index=0, thickness=np.inf, radius=np.inf, conic=0, material=air)
#         self.lens.add_surface(index=1, thickness=10, radius=150, conic=0, material=mat)
#         self.lens.add_surface(index=2, thickness=140, radius=-250, conic=0, material=air, is_stop=True)
#         self.lens.add_surface(index=3)

#         self.lens.add_wavelength(0.55, is_primary=True)

#     def test_f1(self):
#         self.assertAlmostEquals(self.lens.paraxial.f1(), -182.358953, places=5)

#     def test_f2(self):
#         self.assertAlmostEquals(self.lens.paraxial.f2(), 182.358953, places=5)

#     def test_EPD(self):
#         self.assertAlmostEquals(self.lens.paraxial.EPD(), 10.0, places=5)

#     def test_EPL(self):
#         self.assertAlmostEquals(self.lens.paraxial.EPL(), 6.738752, places=5)

#     def test_XPD(self):
#         self.assertAlmostEquals(self.lens.paraxial.XPD(), 9.772357, places=5)

#     def test_XPL(self):
#         self.assertAlmostEquals(self.lens.paraxial.XPL(), -140.0, places=5)

#     def test_magnification(self):
#         self.assertAlmostEqual(self.lens.paraxial.magnification(), 0, places=5)

#     def test_F1(self):
#         self.assertAlmostEqual(self.lens.paraxial.F1(), -179.868184, places=5)

#     def test_F2(self):
#         self.assertAlmostEquals(self.lens.paraxial.F2(), 38.207672, places=5)

#     def test_P1(self):
#         self.assertAlmostEquals(self.lens.paraxial.P1(), 2.490769, places=5)

#     def test_P2(self):
#         self.assertAlmostEquals(self.lens.paraxial.P2(), -144.151281, places=5)

#     def test_N1(self):
#         self.assertAlmostEquals(self.lens.paraxial.N1(), 2.490769, places=5)

#     def test_N2(self):
#         self.assertAlmostEquals(self.lens.paraxial.N2(), -144.151281, places=5)

#     def test_invariant(self):
#         self.assertAlmostEqual(self.lens.paraxial.invariant(), -0.8816349035423249, places=5)

#     def test_marginal_ray(self):
#         y, u = self.lens.paraxial.marginal_ray()
#         y_target = np.array([[5.], [5.], [4.8861783], [1.04759517]])
#         self.assertTrue(np.max(np.abs(y - y_target)) < 0.00001)
#         u_target = np.array([[0.], [-0.01138217], [-0.02741845], [-0.02741845]])
#         self.assertTrue(np.max(np.abs(u - u_target)) < 0.00001)

#     def test_chief_ray(self):
#         y, u = self.lens.paraxial.chief_ray()
#         y_target = np.array([[-1.18822385], [-1.18822385], [0], [25.26082326]])
#         self.assertTrue(np.max(np.abs(y - y_target)) < 0.00001)
#         u_target = np.array([[0.17632698], [0.11882239], [0.18043445], [0.18043445]])
#         self.assertTrue(np.max(np.abs(u - u_target)) < 0.00001)
