import pytest
import numpy as np
from optiland import solves
from optiland.samples.objectives import CookeTriplet


class TestMarginalRayHeightSolve:
    def test_marginal_ray_height_solve_constructor(self):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        solve = solves.MarginalRayHeightSolve(optic, surface_idx, height)

        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_marginal_ray_height_solve_apply(self):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        ya, ua = optic.paraxial.marginal_ray()
        offset = (height - ya[surface_idx]) / ua[surface_idx]
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = np.copy(surf.geometry.cs.z)

        solve = solves.MarginalRayHeightSolve(optic, surface_idx, height)
        solve.apply()

        # Check that surface has been shifted
        assert surf.geometry.cs.z == pytest.approx(z_orig + offset)

        # Check that marginal ray height is correct on surface
        ya, ua = optic.paraxial.marginal_ray()
        assert ya[surface_idx] == pytest.approx(height)


class TestSolveFactory:
    def test_create_solve(self):
        optic = CookeTriplet()
        solve_type = 'marginal_ray_height'
        surface_idx = 7
        height = 0.5

        solve = solves.SolveFactory.create_solve(optic, solve_type,
                                                 surface_idx, height)

        assert isinstance(solve, solves.MarginalRayHeightSolve)
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_create_solve_invalid_solve_type(self):
        optic = CookeTriplet()
        solve_type = 'invalid'
        surface_idx = 7
        height = 0.5

        with pytest.raises(ValueError):
            solves.SolveFactory.create_solve(optic, solve_type,
                                             surface_idx, height)
