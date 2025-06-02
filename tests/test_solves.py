import optiland.backend as be
import pytest

from optiland.solves import (
    BaseSolve,
    MarginalRayHeightSolve,
    ChiefRayHeightSolve,
    QuickFocusSolve,
    SolveFactory,
    SolveManager,
)
from optiland.samples.objectives import CookeTriplet
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
        assert isinstance(solve, MarginalRayHeightSolve) # Use imported class

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

    def test_chief_ray_height_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        surface_idx = 7
        height = 0.2

        yc, uc = optic.paraxial.chief_ray()
        # Ensure uc[surface_idx] is not zero to avoid division by zero
        if be.isclose(uc[surface_idx], 0.0):
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

        # We use BaseSolve.from_dict which relies on the registry populated by __init_subclass__
        # If factory doesn't use registry, this test might need SolveFactory or direct instantiation
        # For now, assuming BaseSolve.from_dict still works due to solves being imported.
        # If SolveFactory.create_solve is the new canonical way, this should change.
        # The task description for factory implies it does not use registry.
        # Let's use SolveFactory.create_solve if BaseSolve.from_dict is no longer the intended path.
        # The original `solves.py` had BaseSolve.from_dict, and `BaseSolve` itself still has it.
        # The `_solve_map` in factory is a parallel mechanism.
        # Sticking to `BaseSolve.from_dict` as per existing test patterns for other solves.
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
        solve = QuickFocusSolve(optic) # Use imported class

        assert solve.optic == optic

    def test_quick_focus_solve_apply(self, set_test_backend):
        optic = CookeTriplet()
        optic.surface_group.surfaces[-1].geometry.cs.z = optic.surface_group.surfaces[
            -1
        ].geometry.cs.z - be.array(10)
        thickness = 42.21812063592369
        solve = QuickFocusSolve(optic) # Use imported class
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1][0]
        pos1 = optic.surface_group.positions[-2][0]

        assert_allclose(pos2 - pos1, thickness, rtol=1e-3)

        # Implementing the extreme shift case.
        optic.surface_group.surfaces[-1].geometry.cs.z += 1000
        solve = QuickFocusSolve(optic) # Use imported class
        solve.apply()

        # Check that surface has been shifted
        pos2 = optic.surface_group.positions[-1][0]
        pos1 = optic.surface_group.positions[-2][0]

        assert_allclose(pos2 - pos1, thickness, rtol=1e-3)


class TestSolveFactory:
    @pytest.mark.parametrize(
        "solve_type_str, expected_class, solve_args, expected_attrs",
        [
            (
                "marginal_ray_height",
                MarginalRayHeightSolve,
                (7, 0.5), # surface_idx, height
                {"surface_idx": 7, "height": 0.5},
            ),
            (
                "chief_ray_height",
                ChiefRayHeightSolve,
                (6, 0.1), # surface_idx, height
                {"surface_idx": 6, "height": 0.1},
            ),
            (
                "quick_focus",
                QuickFocusSolve,
                (None,), # surface_idx is formally passed but ignored by QuickFocusSolve constructor in factory
                {}, # QuickFocusSolve specific attributes not easily checked without applying
            ),
        ],
    )
    def test_create_solve(
        self, set_test_backend, solve_type_str, expected_class, solve_args, expected_attrs
    ):
        optic = CookeTriplet()
        # For height solves, surface_idx and height are actual arguments.
        # For quick_focus, surface_idx is passed to create_solve but ignored by the factory for that type.
        # args are unpacked for height solves.
        
        # surface_idx is the second arg to create_solve, height (or other args) are *args
        s_idx = solve_args[0]
        remaining_args = solve_args[1:] if len(solve_args) > 1 else []


        solve = SolveFactory.create_solve(optic, solve_type_str, s_idx, *remaining_args)

        assert isinstance(solve, expected_class)
        assert solve.optic == optic
        for attr, value in expected_attrs.items():
            assert getattr(solve, attr) == value

    def test_create_solve_invalid_solve_type(self, set_test_backend):
        optic = CookeTriplet()
        solve_type = "invalid_solve_type" # More descriptive
        surface_idx = 7
        height = 0.5 # This arg is not always used, but fine for testing invalid type

        with pytest.raises(ValueError, match=f"Unknown solve type: {solve_type}"): # Check error message
            SolveFactory.create_solve(optic, solve_type, surface_idx, height)

    def test_create_solve_missing_args(self, set_test_backend):
        optic = CookeTriplet()
        with pytest.raises(ValueError, match="Missing 'height' argument"):
            SolveFactory.create_solve(optic, "marginal_ray_height", 7) # No height arg


class TestSolveManager:
    def test_solve_manager_constructor(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic) # Use imported class

        assert manager.optic == optic
        assert len(manager) == 0

    def test_add_solve(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic) # Use imported class
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        manager.add(solve_type, surface_idx, height)

        assert len(manager) == 1
        assert isinstance(manager.solves[0], MarginalRayHeightSolve) # Use imported class
        assert manager.solves[0].optic == optic
        assert manager.solves[0].surface_idx == surface_idx
        assert manager.solves[0].height == height

    def test_apply_solves(self, set_test_backend):
        optic = CookeTriplet()
        manager = SolveManager(optic) # Use imported class
        manager = SolveManager(optic) # Use imported class
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
        manager = SolveManager(optic) # Use imported class, (already correct)
        solve_type = "marginal_ray_height"
        surface_idx = 7
        height = 0.5

        # This was the line with the error in the previous output, but it was a red herring.
        # The previous test output showed:
        # manager = solves.SolveManager(optic)
        # E       NameError: name 'solves' is not defined
        # tests/test_solves.py:288: NameError
        # My previous diff corrected this to "manager = SolveManager(optic)".
        # The error must be from a version of the file before my previous diff.
        # The current version of the file shown in the previous `read_files` or applied diff already has this corrected.
        # I will re-verify the context of the NameError from the previous run.
        # Ah, the previous run's output for test_clear_solves was:
        # >       manager = solves.SolveManager(optic)
        # E       NameError: name 'solves' is not defined
        # tests/test_solves.py:288: NameError
        # And my previous diff for TestSolveManager was:
        # class TestSolveManager:
        #    def test_solve_manager_constructor(self, set_test_backend):
        #        optic = CookeTriplet()
        #        manager = SolveManager(optic) # Use imported class
        # This means the NameError was in test_clear_solves specifically and I missed it in the diff.
        # It should be:
        # manager = SolveManager(optic) # This is already what the current code has from previous diff.

        # Let's re-examine the test output's line numbers if possible or assume the previous diff was incomplete for this specific spot.
        # The diff I provided for the previous step:
        # -        manager = solves.SolveManager(optic)
        # +        manager = SolveManager(optic) # Use imported class
        # This correction was applied to test_solve_manager_constructor, test_add_solve, test_apply_solves.
        # It seems I missed it for test_clear_solves.

        # Correcting it now:
        manager.add(solve_type, surface_idx, height)
        manager.clear()

        assert len(manager) == 0
