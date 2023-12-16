import unittest
import numpy as np
from pyoptic import optic


# TODO: add same test for singlet
class TestDoubleGauss(unittest.TestCase):
    def setUp(self):
        self.lens = optic.Optic()

        self.lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.lens.add_surface(index=1, radius=56.20238, thickness=8.75, material='N-SSK2')
        self.lens.add_surface(index=2, radius=152.28580, thickness=0.5)
        self.lens.add_surface(index=3, radius=37.68262, thickness=12.5, material='N-SK2')
        self.lens.add_surface(index=4, radius=np.inf, thickness=3.8, material=('F5', 'schott'))
        self.lens.add_surface(index=5, radius=24.23130, thickness=16.369445)
        self.lens.add_surface(index=6, radius=np.inf, thickness=13.747957, is_stop=True)
        self.lens.add_surface(index=7, radius=-28.37731, thickness=3.8, material=('F5', 'schott'))
        self.lens.add_surface(index=8, radius=np.inf, thickness=11, material='N-SK16')
        self.lens.add_surface(index=9, radius=-37.92546, thickness=0.5)
        self.lens.add_surface(index=10, radius=177.41176, thickness=7, material='N-SK16')
        self.lens.add_surface(index=11, radius=-79.41143, thickness=61.487536)
        self.lens.add_surface(index=12)

        # add aperture
        self.lens.set_aperture(aperture_type='imageFNO', value=5)
        self.lens.set_field_type(field_type='angle')

        # add field
        self.lens.field_type = 'angle'
        self.lens.add_field(y=0)
        self.lens.add_field(y=10)
        self.lens.add_field(y=14)

        # add wavelength
        self.lens.add_wavelength(value=0.4861)
        self.lens.add_wavelength(value=0.5876, is_primary=True)
        self.lens.add_wavelength(value=0.6563)

        self.lens.update_paraxial()

    def test_seidels(self):
        S = self.lens.aberrations.seidels()
        self.assertAlmostEqual(S[0], -0.003929457875534847, places=9)
        self.assertAlmostEqual(S[1], 0.0003954597633218682, places=9)
        self.assertAlmostEqual(S[2], 0.0034239055031729947, places=9)
        self.assertAlmostEqual(S[3], -0.016264753735226404, places=9)
        self.assertAlmostEqual(S[4], -0.046484107476755930, places=9)

    def test_third_order(self):
        # TODO: understand deltas against nominal values for TAC, AC, color abs
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = self.lens.aberrations.third_order()
        self.assertAlmostEqual(np.sum(TSC), -0.01964728937767421, places=9)
        self.assertAlmostEqual(np.sum(SC), -0.19647289377674193, places=9)
        self.assertAlmostEqual(np.sum(CC), 0.0019772988166093623, places=9)
        self.assertAlmostEqual(np.sum(TCC), 0.005931896449828042, places=9)
        self.assertAlmostEqual(np.sum(TAC), 0.017119527515864978, places=9)  # NOK
        self.assertAlmostEqual(np.sum(AC), 0.17119527515864985, places=9)  # NOK
        self.assertAlmostEqual(np.sum(TPC), -0.08132376867613199, places=9)
        self.assertAlmostEqual(np.sum(PC), -0.8132376867613212, places=9)
        self.assertAlmostEqual(np.sum(DC), -0.2324205373837797, places=9)
        self.assertAlmostEqual(np.sum(TAchC), 0.0295705512988189, places=9)  # NOK
        self.assertAlmostEqual(np.sum(LchC), 0.2957055129881888, places=9)  # NOK
        self.assertAlmostEqual(np.sum(TchC), -0.01804376318260833, places=9)  # NOK
        self.assertAlmostEqual(S[0], -0.003929457875534847, places=9)
        self.assertAlmostEqual(S[1], 0.0003954597633218682, places=9)
        self.assertAlmostEqual(S[2], 0.0034239055031729947, places=9)
        self.assertAlmostEqual(S[3], -0.016264753735226404, places=9)
        self.assertAlmostEqual(S[4], -0.046484107476755930, places=9)
