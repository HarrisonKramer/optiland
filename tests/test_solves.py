import numpy as np
import pytest

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

    def test_to_dict(self):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        solve = solves.MarginalRayHeightSolve(optic, surface_idx, height)
        data = solve.to_dict()

        assert data["type"] == "MarginalRayHeightSolve"
        assert data["surface_idx"] == surface_idx
        assert data["height"] == height

    def test_from_dict(self):
        optic = CookeTriplet()
        data = {"type": "MarginalRayHeightSolve", "surface_idx": 7, "height": 0.5}

        solve = solves.BaseSolve.from_dict(optic, data)

        assert solve.surface_idx == data["surface_idx"]
        assert solve.height == data["height"]

    def test_from_dict_invalid_type(self):
        optic = CookeTriplet()
        data = {"type": "Invalid", "surface_idx": 7, "height": 0.5}

        with pytest.raises(ValueError):
            solves.BaseSolve.from_dict(optic, data)


class TestQuickfocusSolve:
    def test_quick_focus_solve_constructor(self):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z -= 10
        solve = solves.QuickFocusSolve(optic)

        assert solve.optic == optic

    def test_quick_focus_solve_apply(self):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z -= 10
        thickness = 42.21812063592369
        solve = solves.QuickFocusSolve(optic)
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1].astype(float)[0]
        pos1 = optic.surface_group.positions[-2].astype(float)[0]

        assert pos2 - pos1 == pytest.approx(thickness, rel=1e-3)

        # Implementing the extreme shift case.
        optic.surface_group.surfaces[-1].geometry.cs.z += 1000
        solve = solves.QuickFocusSolve(optic)
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1].astype(float)[0]
        pos1 = optic.surface_group.positions[-2].astype(float)[0]

        assert pos2 - pos1 == pytest.approx(thickness, rel=1e-3)


class TestSolveFactory:
    def test_create_solve(self):
        optic = CookeTriplet()
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        solve = solves.SolveFactory.create_solve(optic, solve_type, surface_idx, height)

        assert isinstance(solve, solves.MarginalRayHeightSolve)
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_create_solve_invalid_solve_type(self):
        optic = CookeTriplet()
        solve_type = "invalid"
        surface_idx = 7
        height = 0.5

        with pytest.raises(ValueError):
            solves.SolveFactory.create_solve(optic, solve_type, surface_idx, height)


class TestSolveManager:
    def test_solve_manager_constructor(self):
        optic = CookeTriplet()
        manager = solves.SolveManager(optic)

        assert manager.optic == optic
        assert len(manager) == 0

    def test_add_solve(self):
        optic = CookeTriplet()
        manager = solves.SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        manager.add(solve_type, surface_idx, height)

        assert len(manager) == 1
        assert isinstance(manager.solves[0], solves.MarginalRayHeightSolve)
        assert manager.solves[0].optic == optic
        assert manager.solves[0].surface_idx == surface_idx
        assert manager.solves[0].height == height

    def test_apply_solves(self):
        optic = CookeTriplet()
        manager = solves.SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        ya, ua = optic.paraxial.marginal_ray()
        offset = (height - ya[surface_idx]) / ua[surface_idx]
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = np.copy(surf.geometry.cs.z)

        manager.add(solve_type, surface_idx, height)
        manager.apply()

        # Check that surface has been shifted
        assert surf.geometry.cs.z == pytest.approx(z_orig + offset)

        # Check that marginal ray height is correct on surface
        ya, ua = optic.paraxial.marginal_ray()
        assert ya[surface_idx] == pytest.approx(height)

    def test_clear_solves(self):
        optic = CookeTriplet()
        manager = solves.SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        manager.add(solve_type, surface_idx, height)
        manager.clear()

        assert len(manager) == 0
