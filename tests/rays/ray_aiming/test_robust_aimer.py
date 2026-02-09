
from __future__ import annotations

import unittest
import numpy as np
import optiland.backend as be
from optiland.materials.ideal import IdealMaterial
from optiland.optic import Optic
from optiland.rays.ray_aiming.robust import RobustRayAimer

class TestRobustRayAimer(unittest.TestCase):
    def setUp(self):
        be.set_backend("numpy")
        self.optic = Optic()
        
        # Wide angle or highly aberrated system to justify robust aiming
        # Simple setup for now: Aperture stop at surface 2
        self.optic.add_surface(index=0, thickness=np.inf)
        self.optic.add_surface(
            index=1, radius=20, thickness=10, material=IdealMaterial(n=1.5)
        )
        self.optic.add_surface(index=2, is_stop=True, aperture=10.0, thickness=10)
        self.optic.add_surface(index=3)

        self.optic.set_field_type("angle")
        self.optic.add_field(y=0)
        self.optic.add_field(y=20) # 20 degrees
        self.optic.set_aperture("float_by_stop_size", 1.0)
        self.optic.add_wavelength(0.55)

    def test_robust_aimer_initialization(self):
        aimer = RobustRayAimer(self.optic)
        self.assertEqual(aimer.scale_fields, True)

    def test_integration_via_optic(self):
        # Set robust mode via optic
        self.optic.set_ray_aiming("robust", scale_fields=True)
        self.assertEqual(self.optic.ray_aiming_config["mode"], "robust")
        
        # Generate rays using RayGenerator (implicitly created/updated)
        # We need to access the generator or manually create one
        from optiland.rays import RayGenerator
        gen = RayGenerator(self.optic)
        
        # Trigger generation (and thus aimer update)
        rays = gen.generate_rays(0, 1, 0, 1, 0.55) # Hx, Hy, Px, Py, wav
        
        # Verify the generator is using RobustRayAimer
        self.assertIsInstance(gen.aimer, RobustRayAimer)
        self.assertEqual(gen.aimer.scale_fields, True)

    def test_aiming_execution(self):
        # Directly test allow execution logic
        aimer = RobustRayAimer(self.optic)
        fields = (0, 1) # Full field (20 deg)
        wavelengths = 0.55
        pupil = (0, 1) # Marginal ray
        
        # Expect successful run
        x, y, z, L, M, N = aimer.aim_rays(fields, wavelengths, pupil)
        
        # Verify we hit the stop (at least reasonably close)
        # Trace result
        from optiland.rays import RealRays
        rays = RealRays(
            x, y, z, L, M, N, intensity=be.ones_like(x), wavelength=wavelengths
        )
        for i in range(self.optic.surface_group.stop_index + 1):
             self.optic.surface_group.surfaces[i].trace(rays)
        
        stop_r = self.optic.surface_group.surfaces[2].aperture.r_max
        target_y = stop_r
        error = np.abs(rays.y - target_y)
        
        # Check convergence
        self.assertTrue(np.all(error < 1e-4))

if __name__ == "__main__":
    unittest.main()
