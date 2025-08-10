import pytest

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.solves import (
    BaseSolve,
    ChiefRayHeightSolve,
    MarginalRayHeightSolve,
    QuickFocusSolve,
    SolveFactory,
    SolveManager,
)

from .utils import assert_allclose


class TestMarginalRayHeightSolve:
    def test_marginal_ray_height_solve_constructor(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        solve = MarginalRayHeightSolve(optic, surface_idx, height)

        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_marginal_ray_height_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        ya, ua = optic.paraxial.marginal_ray()
        offset = (height - ya[surface_idx]) / ua[surface_idx]
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = be.copy(surf.geometry.cs.z)

        solve = MarginalRayHeightSolve(optic, surface_idx, height)
        solve.apply()

        # Check that surface has been shifted
        assert_allclose(surf.geometry.cs.z, z_orig + offset)

        # Check that marginal ray height is correct on surface
        ya, ua = optic.paraxial.marginal_ray()
        assert_allclose(ya[surface_idx], height)

    def test_to_dict(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        solve = MarginalRayHeightSolve(optic, surface_idx, height)
        data = solve.to_dict()

        assert data["type"] == "MarginalRayHeightSolve"
        assert data["surface_idx"] == surface_idx
        assert data["height"] == height

    def test_from_dict(self, set_test_backend):
        optic = CookeTriplet()
        data = {"type": "MarginalRayHeightSolve", "surface_idx": 7, "height": 0.5}

        solve = BaseSolve.from_dict(optic, data) # Use imported BaseSolve

        assert solve.surface_idx == data["surface_idx"]
        assert solve.height == data["height"]
        assert isinstance(solve, MarginalRayHeightSolve)

    def test_from_dict_invalid_type(self, set_test_backend):
        optic = CookeTriplet()
        data = {"type": "Invalid", "surface_idx": 7, "height": 0.5}

        with pytest.raises(ValueError):
            BaseSolve.from_dict(optic, data) # Changed to use imported BaseSolve


class TestChiefRayHeightSolve:
    def test_chief_ray_height_solve_constructor(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7  # Example surface index
        height = 0.2  # Example target chief ray height

        solve = ChiefRayHeightSolve(optic, surface_idx, height)

        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_invalid_surface_idx(self, set_test_backend):
        with pytest.raises(ValueError):
            optic = CookeTriplet()
            surface_idx = None
            height = 0.2

            solve = ChiefRayHeightSolve(optic, surface_idx, height)

    def test_chief_ray_height_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.2

        yc, uc = optic.paraxial.chief_ray()
        # Ensure uc[surface_idx] is not zero to avoid division by zero
        if be.abs(uc[surface_idx] - 0.0) < 1e-6:
            pytest.skip("Chief ray slope is zero at the specified surface, cannot apply solve.")

        offset = (height - yc[surface_idx]) / uc[surface_idx]
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = be.copy(surf.geometry.cs.z)

        solve = ChiefRayHeightSolve(optic, surface_idx, height)
        solve.apply()

        # Check that surface has been shifted
        assert_allclose(surf.geometry.cs.z, z_orig + offset, atol=1e-7)

        # Check that chief ray height is correct on surface
        yc_new, _ = optic.paraxial.chief_ray()
        assert_allclose(yc_new[surface_idx], height, atol=1e-7)

    def test_to_dict(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.2

        solve = ChiefRayHeightSolve(optic, surface_idx, height)
        data = solve.to_dict()

        assert data["type"] == "ChiefRayHeightSolve"
        assert data["surface_idx"] == surface_idx
        assert data["height"] == height

    def test_from_dict(self, set_test_backend):
        optic = CookeTriplet()
        data = {"type": "ChiefRayHeightSolve", "surface_idx": 7, "height": 0.2}
        solve = BaseSolve.from_dict(optic, data)

        assert isinstance(solve, ChiefRayHeightSolve)
        assert solve.surface_idx == data["surface_idx"]
        assert solve.height == data["height"]


class TestQuickfocusSolve:
    def test_quick_focus_solve_constructor(self, set_test_backend):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z = optic.surface_group.surfaces[
            -1
        ].geometry.cs.z - be.array(10)
        solve = QuickFocusSolve(optic)

        assert solve.optic == optic

    def test_quick_focus_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z = optic.surface_group.surfaces[
            -1
        ].geometry.cs.z - be.array(10)
        thickness = 42.21812063592369
        solve = QuickFocusSolve(optic)
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1][0]
        pos1 = optic.surface_group.positions[-2][0]

        assert_allclose(pos2 - pos1, thickness, rtol=1e-3)

        # Implementing the extreme shift case.
        optic.surface_group.surfaces[-1].geometry.cs.z += 1000
        solve = QuickFocusSolve(optic)
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1][0]
        pos1 = optic.surface_group.positions[-2][0]

        assert_allclose(pos2 - pos1, thickness, rtol=1e-3)


class TestSolveFactory:
    def test_create_solve(self, set_test_backend):
        optic = CookeTriplet()
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        solve = SolveFactory.create_solve(optic, solve_type, surface_idx, height)

        assert isinstance(solve, MarginalRayHeightSolve)
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

        solve_type = "chief_ray_height"
        surface_idx = 6
        height = 0.5

        solve = SolveFactory.create_solve(optic, solve_type, surface_idx, height)

        assert isinstance(solve, ChiefRayHeightSolve)
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height

    def test_create_solve_invalid_solve_type(self, set_test_backend):
        optic = CookeTriplet()
        solve_type = "invalid"
        surface_idx = 7
        height = 0.5

        with pytest.raises(ValueError):
            SolveFactory.create_solve(optic, solve_type, surface_idx, height)


class TestSolveManager:
    def test_solve_manager_constructor(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic)

        assert manager.optic == optic
        assert len(manager) == 0

    def test_add_solve(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        manager.add(solve_type, surface_idx, height)

        assert len(manager) == 1
        assert isinstance(manager.solves[0], MarginalRayHeightSolve)
        assert manager.solves[0].optic == optic
        assert manager.solves[0].surface_idx == surface_idx
        assert manager.solves[0].height == height

    def test_apply_solves(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic)
        manager = SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        ya, ua = optic.paraxial.marginal_ray()
        offset = (height - ya[surface_idx]) / ua[surface_idx]
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = be.copy(surf.geometry.cs.z)

        manager.add(solve_type, surface_idx, height)
        manager.apply()

        # Check that surface has been shifted
        assert_allclose(surf.geometry.cs.z, z_orig + offset)

        # Check that marginal ray height is correct on surface
        ya, ua = optic.paraxial.marginal_ray()
        assert_allclose(ya[surface_idx], height)

    def test_clear_solves(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        manager.add(solve_type, surface_idx, height)
        manager.clear()

        assert len(manager) == 0
