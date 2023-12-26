import unittest
import numpy as np
from optiland import analysis, optic


class TestSingletObjInfinityStop1(unittest.TestCase):
    def setUp(self):
        self.lens = optic.Optic()

        self.lens.surface_group.surfaces = []

        self.lens.add_surface(index=0, radius=np.inf, thickness=np.inf)
        self.lens.add_surface(index=1, radius=22.01359, thickness=3.25896,
                              material='SK16')
        self.lens.add_surface(index=2, radius=-435.76044, thickness=6.00755)
        self.lens.add_surface(index=3, radius=-22.21328, thickness=0.99997,
                              material=('F2', 'schott'))
        self.lens.add_surface(index=4, radius=20.29192, thickness=4.75041,
                              is_stop=True)
        self.lens.add_surface(index=5, radius=79.68360, thickness=2.95208,
                              material='SK16')
        self.lens.add_surface(index=6, radius=-18.39533, thickness=42.20778)
        self.lens.add_surface(index=7)

        # add aperture
        self.lens.set_aperture(aperture_type='EPD', value=10)

        # add field
        self.lens.set_field_type(field_type='angle')
        self.lens.add_field(y=0)
        self.lens.add_field(y=14)
        self.lens.add_field(y=20)

        # add wavelength
        self.lens.add_wavelength(value=0.48)
        self.lens.add_wavelength(value=0.55, is_primary=True)
        self.lens.add_wavelength(value=0.65)

        self.lens.update_paraxial()

    def test_spot_geometric_radius(self):
        spot = analysis.SpotDiagram(self.lens)
        geo_radius = spot.geometric_spot_radius()

        self.assertAlmostEqual(geo_radius[0][0], 0.00597244087781, places=9)
        self.assertAlmostEqual(geo_radius[0][1], 0.00628645771124, places=9)
        self.assertAlmostEqual(geo_radius[0][2], 0.00931911440064, places=9)

        self.assertAlmostEqual(geo_radius[1][0], 0.03717783072826, places=9)
        self.assertAlmostEqual(geo_radius[1][1], 0.03864613392848, places=9)
        self.assertAlmostEqual(geo_radius[1][2], 0.04561512437816, places=9)

        self.assertAlmostEqual(geo_radius[2][0], 0.01951655430245, places=9)
        self.assertAlmostEqual(geo_radius[2][1], 0.02342659090311, places=9)
        self.assertAlmostEqual(geo_radius[2][2], 0.03747033587405, places=9)

    def test_spot_rms_radius(self):
        spot = analysis.SpotDiagram(self.lens)
        rms_radius = spot.rms_spot_radius()

        self.assertAlmostEqual(rms_radius[0][0], 0.003791335461448, places=9)
        self.assertAlmostEqual(rms_radius[0][1], 0.004293689564257, places=9)
        self.assertAlmostEqual(rms_radius[0][2], 0.006195618755672, places=9)

        self.assertAlmostEqual(rms_radius[1][0], 0.015694600107671, places=9)
        self.assertAlmostEqual(rms_radius[1][1], 0.016786721284464, places=9)
        self.assertAlmostEqual(rms_radius[1][2], 0.019109151416248, places=9)

        self.assertAlmostEqual(rms_radius[2][0], 0.013229165357157, places=9)
        self.assertAlmostEqual(rms_radius[2][1], 0.012081348897953, places=9)
        self.assertAlmostEqual(rms_radius[2][2], 0.013596802321537, places=9)

    def test_ray_fan(self):
        fan = analysis.RayFan(self.lens)

        self.assertEqual(fan.data['Px'][0], -1)
        self.assertEqual(fan.data['Px'][-1], 1)

        self.assertEqual(fan.data['Py'][0], -1)
        self.assertEqual(fan.data['Py'][-1], 1)

        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.48']['x'][0],
                               0.00230694465588267, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.48']['x'][-1],
                               -0.0024693549632838, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.48']['y'][0],
                               0.00230694465588267, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.48']['y'][-1],
                               -0.0024693549632838, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.55']['x'][0],
                               0.00411447192762305, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.55']['x'][-1],
                               -0.0042768822350241, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.55']['y'][0],
                               0.00411447192762305, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.55']['y'][-1],
                               -0.0042768822350241, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.65']['x'][0],
                               -8.948985061928e-05, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.65']['x'][-1],
                               -7.292045678185e-05, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.65']['y'][0],
                               -8.948985061928e-05, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.0)']['0.65']['y'][-1],
                               -7.292045678185e-05, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.48']['x'][0],
                               0.01976688587767077, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.48']['x'][-1],
                               -0.0196959560263178, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.48']['y'][0],
                               -0.0232699816183430, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.48']['y'][-1],
                               0.03922178177355029, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.55']['x'][0],
                               0.02145565610521620, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.55']['x'][-1],
                               -0.0213847262538632, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.55']['y'][0],
                               -0.0248752380425489, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.55']['y'][-1],
                               0.04069008497376458, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.65']['x'][0],
                               0.01706095214297792, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.65']['x'][-1],
                               -0.0169900222916250, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.65']['y'][0],
                               -0.0323595284535702, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 0.7)']['0.65']['y'][-1],
                               0.04765907542344116, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.48']['x'][0],
                               0.01571814948112706, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.48']['x'][-1],
                               -0.0155594842298404, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.48']['y'][0],
                               -0.0043495236774511, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.48']['y'][-1],
                               0.01314983854687312, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.55']['x'][0],
                               0.01701576639943294, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.55']['x'][-1],
                               -0.0168571011481462, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.55']['y'][0],
                               -0.0169019565813819, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.55']['y'][-1],
                               0.02265130085670819, places=9)

        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.65']['x'][0],
                               0.01222467864771505, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.65']['x'][-1],
                               -0.0120660133964284, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.65']['y'][0],
                               -0.0338080841047059, places=9)
        self.assertAlmostEqual(fan.data['(0.0, 1.0)']['0.65']['y'][-1],
                               0.03669504582764205, places=9)
