import pytest
import numpy as np

import optiland.backend as be
from optiland.samples.objectives import CookeTriplet
from optiland.solves import (
    BaseSolve,
    ThicknessSolve,
    MarginalRayHeightSolve,
    ChiefRayHeightSolve,
    CurvatureSolve,
    MarginalRayAngleSolve,
    ChiefRayAngleSolve,
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
        assert isinstance(solve, ThicknessSolve)

    def test_marginal_ray_height_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.5

        # Paraxial trace
        ya, ua = optic.paraxial.marginal_ray()
        
        # Incident slope is ua[surface_idx-1] (or ua[0] if surface_idx=0)
        u_incident = ua[0] if surface_idx == 0 else ua[surface_idx - 1]
        
        offset = (height - ya[surface_idx]) / u_incident
        
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

        solve = BaseSolve.from_dict(optic, data)

        assert solve.surface_idx == data["surface_idx"]
        assert solve.height == data["height"]
        assert isinstance(solve, MarginalRayHeightSolve)


class TestChiefRayHeightSolve:
    def test_chief_ray_height_solve_constructor(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.2

        solve = ChiefRayHeightSolve(optic, surface_idx, height)

        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.height == height
        assert isinstance(solve, ThicknessSolve)

    def test_chief_ray_height_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.2

        yc, uc = optic.paraxial.chief_ray()
        
        # Incident slope
        u_incident = uc[0] if surface_idx == 0 else uc[surface_idx - 1]

        if be.abs(u_incident - 0.0) < 1e-6:
            pytest.skip(
                "Chief ray slope is zero at the specified surface, cannot apply solve."
            )

        offset = (height - yc[surface_idx]) / u_incident
        surf = optic.surface_group.surfaces[surface_idx]
        z_orig = be.copy(surf.geometry.cs.z)

        solve = ChiefRayHeightSolve(optic, surface_idx, height)
        solve.apply()

        # Check that surface has been shifted
        assert_allclose(surf.geometry.cs.z, z_orig + offset, atol=1e-7)

        # Check that chief ray height is correct on surface
        yc_new, _ = optic.paraxial.chief_ray()
        assert_allclose(yc_new[surface_idx], height, atol=1e-7)


class TestMarginalRayAngleSolve:
    def test_constructor(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 3
        angle = -0.05
        
        solve = MarginalRayAngleSolve(optic, surface_idx, angle)
        
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.angle == angle
        assert isinstance(solve, CurvatureSolve)

    def test_apply(self, set_test_backend):
        optic = CookeTriplet()
        # Choose a curved surface, e.g., surface 1 (first lens front)
        surface_idx = 1
        target_angle = -0.1
        
        # Initial state
        y, u = optic.paraxial.marginal_ray()
        
        # Target angle is the EXIT slope u[surface_idx] (slope after surface_idx)
        # u[i] output from paraxial trace corresponds to slope AFTER surface i.
        # u[surface_idx] is the slope we modified.
        if surface_idx < len(u):
            assert not be.isclose(u[surface_idx], target_angle, atol=1e-4)
        
        # Apply solve
        solve = MarginalRayAngleSolve(optic, surface_idx, target_angle)
        solve.apply()
        
        # Re-trace and verify
        y_new, u_new = optic.paraxial.marginal_ray()
        
        if surface_idx < len(u_new):
            assert_allclose(u_new[surface_idx], target_angle, atol=1e-4)


class TestChiefRayAngleSolve:
    def test_constructor(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 3
        angle = -0.05
        
        solve = ChiefRayAngleSolve(optic, surface_idx, angle)
        
        assert solve.optic == optic
        assert solve.surface_idx == surface_idx
        assert solve.angle == angle
        assert isinstance(solve, CurvatureSolve)

    def test_apply(self, set_test_backend):
        optic = CookeTriplet()
        # Choose a surface
        surface_idx = 2
        target_angle = 0.4
        
        # Initial state
        y, u = optic.paraxial.chief_ray()
        
        if surface_idx < len(u):
            assert not be.isclose(u[surface_idx], target_angle, atol=1e-4)
        
        # Apply solve
        solve = ChiefRayAngleSolve(optic, surface_idx, target_angle)
        solve.apply()
        
        # Re-trace and verify
        y_new, u_new = optic.paraxial.chief_ray()
        
        if surface_idx < len(u_new):
            assert_allclose(u_new[surface_idx], target_angle, atol=1e-4)


class TestQuickfocusSolve:
    def test_quick_focus_solve_constructor(self, set_test_backend):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z = optic.surface_group.surfaces[-1].geometry.cs.z - be.array(10)
        solve = QuickFocusSolve(optic)
        assert solve.optic == optic

    def test_quick_focus_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z = optic.surface_group.surfaces[-1].geometry.cs.z - be.array(10)
        thickness = 42.21812063592369
        solve = QuickFocusSolve(optic)
        solve.apply()

        pos2 = optic.surface_group.positions[-1][0]
        pos1 = optic.surface_group.positions[-2][0]

        assert_allclose(pos2 - pos1, thickness, rtol=1e-3)


class TestSolveFactory:
    def test_create_solve(self, set_test_backend):
        optic = CookeTriplet()
        solve = SolveFactory.create_solve(optic, "marginal_ray_height", 7, 0.5)
        assert isinstance(solve, MarginalRayHeightSolve)

        solve = SolveFactory.create_solve(optic, "chief_ray_height", 6, 0.5)
        assert isinstance(solve, ChiefRayHeightSolve)

        solve = SolveFactory.create_solve(optic, "marginal_ray_angle", 2, 0.1)
        assert isinstance(solve, MarginalRayAngleSolve)

        solve = SolveFactory.create_solve(optic, "chief_ray_angle", 2, 0.1)
        assert isinstance(solve, ChiefRayAngleSolve)

    def test_create_solve_invalid_solve_type(self, set_test_backend):
        optic = CookeTriplet()
        with pytest.raises(ValueError):
            SolveFactory.create_solve(optic, "invalid", 7, 0.5)


class TestSolveManager:
    def test_solve_manager_basics(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic)
        assert len(manager) == 0
        
        manager.add("marginal_ray_height", 7, 0.5)
        assert len(manager) == 1
        assert isinstance(manager.solves[0], MarginalRayHeightSolve)
        
        manager.clear()
        assert len(manager) == 0
