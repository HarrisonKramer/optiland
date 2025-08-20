import unittest

from optiland.fields.field_solvers import (
    ParaxialImageHeightSolver,
    RealImageHeightSolver,
)

# To mock/stub Optic and related components if actual classes are not available.
# For now, we'll define a simple mock Optic class here.
# In a real scenario, this would come from `optiland.optic` and `optiland.paraxial`.

# --- Mocking/Stubbing Optic and related components ---
# This is a simplified mock for testing purposes.
# It needs to provide the interface expected by the solvers.


class MockParaxialModel:
    # Simplified paraxial model representation.
    # For a lens with focal length f, object at infinity:
    #   y_image = f * tan(angle_y) approx f * angle_y
    # For finite object: magnification m = y_image / y_object
    # chief_ray for object at infinity:
    #   y_image = EFL * field_angle_y (if field_angle_y is in radians)
    # chief_ray for finite object: y_image = magnification * object_height_y
    def __init__(self, efl=None, magnification=None):
        self.efl = efl
        self.magnification = magnification


class MockRay:
    def __init__(self, y_at_image_surface=0.0, success=True):
        self.y_at_image_surface = y_at_image_surface
        self.success = success


class MockOptic:
    def __init__(
        self,
        fields,
        object_is_at_infinity,
        paraxial_efl=None,
        paraxial_magnification=None,
    ):
        self.fields = fields  # e.g., [(0, 0), (0, 10)] for angles in degrees or mm
        self.object_is_at_infinity = object_is_at_infinity
        self.paraxial_model = None  # Will be set by update_paraxial_model
        self._paraxial_efl = paraxial_efl  # Used to create the paraxial model
        self._paraxial_magnification = paraxial_magnification
        self.reference_wavelength = 0.550  # microns, typical reference

        # Attributes for RealFieldSolver's objective_function ray creation
        # These would normally be part of a more complex ray generation system
        # For testing, we can make them simple:
        # Assume real trace matches paraxial for perfect lens
        self.efl_for_real_trace = paraxial_efl
        self.mag_for_real_trace = paraxial_magnification

    def update_paraxial_model(self):
        # This would normally involve complex calculations.
        # For mock, just create the MockParaxialModel.
        self.paraxial_model = MockParaxialModel(
            efl=self._paraxial_efl, magnification=self._paraxial_magnification
        )

    # Mocked paraxial module functions (would normally be in `optiland.paraxial`)
    class paraxial:
        @staticmethod
        def chief_ray(Hy, pm, obj_is_inf):
            # pm is the MockParaxialModel
            # Hy is object field (angle in rad if obj_is_inf, else height)
            if obj_is_inf:
                if pm.efl is None:
                    raise ValueError(
                        "EFL not set for paraxial chief ray (object at infinity)"
                    )
                # Assuming Hy is in radians for this calculation if it's an angle.
                # If Hy from optic.fields is in degrees, it needs conversion.
                # For simplicity, assume Hy passed here is ready for use.
                y_image = pm.efl * Hy
                return 0, 0, y_image, 0  # y_stop, u_stop, y_image, u_image
            # else:
            if pm.magnification is None:
                raise ValueError(
                    "Magnification not set for paraxial chief ray (finite object)"
                )
            y_image = pm.magnification * Hy
            return 0, 0, y_image, 0

    def prepare_ray_for_tracing(self, Hx, Hy, Px, Py, wavelength):
        # Crucial method for RealFieldSolver.
        # Should return MockRay, but properties (esp. initial conditions)
        # would be set based on Hx, Hy, Px, Py.
        # For objective function in RealFieldSolver, only need trace_generic result.
        # So, method can be simple, as trace_generic will determine outcome.
        # Store Hy to be used by trace_generic.
        self._current_Hy_for_real_trace = Hy
        return MockRay()  # Dummy ray object, actual outcome in trace_generic

    def trace_generic(self, ray_obj):
        # Simulates real ray trace for the objective function.
        # Uses Hy stored by prepare_ray_for_tracing.
        Hy = self._current_Hy_for_real_trace

        # Simulate image height based on whether object is at infinity or finite.
        # Mock assumes "perfect lens" behavior for real rays too.
        if self.object_is_at_infinity:
            if self.efl_for_real_trace is None:
                return MockRay(success=False)  # Cannot trace
            # Assume Hy is angle (needs to be radians if fields are degrees).
            # Assume Hy passed in units formula expects (e.g. rad for angle * EFL).
            y_image = self.efl_for_real_trace * Hy
        else:
            if self.mag_for_real_trace is None:
                return MockRay(success=False)  # Cannot trace
            y_image = self.mag_for_real_trace * Hy

        return MockRay(y_at_image_surface=y_image, success=True)


# --- Unit Tests ---


class TestParaxialFieldSolver(unittest.TestCase):
    def test_solve_object_at_infinity_perfect_lens(self):
        # Perfect lens, EFL = 100mm. Object @ infinity. Field is angle.
        # `optic.fields` angles typically in degrees. Solver expects/returns same unit.
        # `paraxial.chief_ray` formula: y_image = EFL * angle_rad.
        # If field angle in degrees, it must be converted for formula.
        # Solver should handle abstraction: input unit -> calc -> output unit.
        # For mock, `Hy` in `chief_ray` needs to be radians if angle.
        # `reference_object_field_y` from `optic.fields` (e.g. degrees).
        # Solver scales this reference field.

        # Assume `optic.fields` stores angles in degrees.
        # EFL = 100mm. Max field = 5 degrees.
        # y_image_ref = 100 * tan(5 deg) approx 100 * (5*pi/180) = 8.726 mm

        # `ParaxialFieldSolver.solve` uses `optic.fields[-1][1]` as ref_obj_field_y
        # then calls `paraxial.chief_ray(ref_obj_field_y, ...)`.
        # If `ref_obj_field_y` is angle in degrees, mock `paraxial.chief_ray`
        # must handle it (e.g. convert to rad if formula uses rad).
        # Current mock `paraxial.chief_ray` directly uses `Hy * EFL`.
        # So, if `optic.fields` in degrees, ensure units consistent.

        # Define mock `paraxial.chief_ray` to expect radians for angles.
        # Solver takes `ref_obj_field_y` (degrees), scales it.
        # Test setup needs care with units.
        # `ParaxialFieldSolver` itself doesn't convert units; it scales
        # the provided field value. The `paraxial.chief_ray` implementation
        # is where units matter for calculation.

        # If `optic.fields` = [(0,0), (0, 5.0)] (degrees)
        # `reference_object_field_y` = 5.0 (degrees)
        # `ref_chief_ray_y_image` = EFL * (5.0 * np.pi/180) for calculation.

        # Simplify mock: assume fields given in units expected by formulas,
        # or solver works with whatever unit is in `optic.fields`.
        # Prompt implies solver returns "object field value" (angle or height).
        # Assume units consistent: if fields in deg, solver returns deg.

        # Test Case: EFL=100. Object @ inf. Max field in `optic.fields` is 1.0 (rad).
        # So `reference_object_field_y` = 1.0 rad.
        # Reference image height = 100 * 1.0 = 100.
        # Target image height = 50.
        # Expected scale factor = 50 / 100 = 0.5.
        # Expected solved field = 1.0 rad * 0.5 = 0.5 rad.

        mock_optic_inf = MockOptic(
            fields=[(0, 0), (0, 1.0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        mock_optic_inf.update_paraxial_model()  # Important to init paraxial_model

        solver = ParaxialFieldSolver()
        target_image_height = 50.0
        # Internally, for Hy=1.0 rad, ref_image_height = 100 * 1.0 = 100.0
        # Scale factor = 50.0 / 100.0 = 0.5
        # Solved field = 1.0 * 0.5 = 0.5
        solved_field = solver.solve(mock_optic_inf, target_image_height)
        self.assertAlmostEqual(solved_field, 0.5, places=7)

        # Test with target_image_height = 0
        solved_field_zero = solver.solve(mock_optic_inf, 0.0)
        self.assertAlmostEqual(solved_field_zero, 0.0, places=7)

    def test_solve_finite_object_perfect_lens(self):
        # Perfect lens, magnification = -2.0 (inverted, magnified). Obj finite.
        # Max field height in `optic.fields` = 10mm.
        # `reference_object_field_y` = 10.0 mm.
        # Reference image height = -2.0 * 10.0 = -20.0 mm.
        # Target image height = -10.0 mm.
        # Expected scale factor = -10.0 / -20.0 = 0.5.
        # Expected solved field = 10.0 mm * 0.5 = 5.0 mm.

        mock_optic_finite = MockOptic(
            fields=[(0, 0), (0, 10.0)],
            object_is_at_infinity=False,
            paraxial_magnification=-2.0,
        )
        mock_optic_finite.update_paraxial_model()

        solver = ParaxialFieldSolver()
        target_image_height = -10.0
        solved_field = solver.solve(mock_optic_finite, target_image_height)
        self.assertAlmostEqual(solved_field, 5.0, places=7)

        # Test with target_image_height = 0
        solved_field_zero = solver.solve(mock_optic_finite, 0.0)
        self.assertAlmostEqual(solved_field_zero, 0.0, places=7)

    def test_solve_reference_field_zero(self):
        mock_optic = MockOptic(
            fields=[(0, 0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        mock_optic.update_paraxial_model()
        solver = ParaxialFieldSolver()

        # Target is 0, ref field is 0, should solve to 0
        self.assertAlmostEqual(solver.solve(mock_optic, 0.0), 0.0, places=7)

        # Target is non-zero, ref field is 0, should raise error
        with self.assertRaisesRegex(ValueError, "Reference object field is zero"):
            solver.solve(mock_optic, 10.0)

    def test_solve_reference_image_height_zero(self):
        # Optic where max field (non-zero) maps to zero image height
        # (e.g. afocal system, or error in setup).
        # Mock `paraxial.chief_ray` to return 0 img height for non-zero input field.
        class MockOpticRefImgZero(MockOptic):
            class paraxial:
                @staticmethod
                def chief_ray(Hy, pm, obj_is_inf):
                    return 0, 0, 0, 0  # Always 0 image height

        mock_optic = MockOpticRefImgZero(
            fields=[(0, 0), (0, 1.0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        # Manually assign custom paraxial sub-class if not done by constructor.
        mock_optic.paraxial = MockOpticRefImgZero.paraxial
        mock_optic.update_paraxial_model()
        solver = ParaxialFieldSolver()

        # Target is 0, ref image height is 0 (for ref_obj_field=1.0). Solves to 0.
        self.assertAlmostEqual(solver.solve(mock_optic, 0.0), 0.0, places=7)

        # Target non-zero, ref image height 0. Should raise error.
        with self.assertRaisesRegex(
            ValueError, "Reference field results in zero image height"
        ):
            solver.solve(mock_optic, 10.0)


class TestRealFieldSolver(unittest.TestCase):
    def test_solve_object_at_infinity_perfect_lens_real(self):
        # Similar to paraxial test, but using RealFieldSolver.
        # Mock Optic's `trace_generic` should behave like a perfect lens.
        # EFL = 100. Fields in radians for calculation simplicity here.
        # `optic.fields` = [(0,-1.0),(0,1.0)] (rad) -> defines solver search range.

        # Objective function: image_y - target_image_height
        # image_y = EFL * object_field_y (radians)
        # Target image_y = 50. EFL = 100.
        # 100 * object_field_y - 50 = 0  => object_field_y = 0.5 radians.

        mock_optic_inf = MockOptic(
            fields=[(0, -1.0), (0, 1.0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        # RealFieldSolver uses `optic.prepare_ray_for_tracing` and
        # `optic.trace_generic`. Mocked in MockOptic to use `efl_for_real_trace`
        # (from `paraxial_efl`). Not strictly needed by RealFieldSolver.
        mock_optic_inf.update_paraxial_model()

        solver = RealFieldSolver()
        target_image_height = 50.0
        solved_field = solver.solve(mock_optic_inf, target_image_height, tol=1e-7)
        self.assertAlmostEqual(solved_field, 0.5, places=6)  # Solver precision

        # Test target_image_height = 0
        solved_field_zero = solver.solve(mock_optic_inf, 0.0, tol=1e-7)
        self.assertAlmostEqual(solved_field_zero, 0.0, places=6)

        # Test negative target image height
        target_image_height_neg = -25.0
        # Expected field = -25.0 / 100.0 = -0.25
        solved_field_neg = solver.solve(
            mock_optic_inf, target_image_height_neg, tol=1e-7
        )
        self.assertAlmostEqual(solved_field_neg, -0.25, places=6)

    def test_solve_finite_object_perfect_lens_real(self):
        # Magnification = -2.0. Fields are object heights.
        # `optic.fields` = [(0, -20.0), (0, 20.0)] (mm)
        # image_y = Mag * object_field_y
        # Target image_y = -10.0. Mag = -2.0.
        # -2.0 * obj_field_y - (-10.0)=0 => -2.0*obj_field_y=-10.0 => obj_field_y=5.0mm

        mock_optic_finite = MockOptic(
            fields=[(0, -20.0), (0, 20.0)],
            object_is_at_infinity=False,
            paraxial_magnification=-2.0,
        )
        mock_optic_finite.update_paraxial_model()

        solver = RealFieldSolver()
        target_image_height = -10.0
        solved_field = solver.solve(mock_optic_finite, target_image_height, tol=1e-7)
        self.assertAlmostEqual(solved_field, 5.0, places=6)

        # Test target_image_height = 0
        solved_field_zero = solver.solve(mock_optic_finite, 0.0, tol=1e-7)
        self.assertAlmostEqual(solved_field_zero, 0.0, places=6)

    def test_solve_real_no_solution_in_range(self):
        # EFL=100. Max field in `optic.fields`=0.1 rad (img height range -10 to 10).
        # Target img height=20 (outside achievable range for bracket).
        mock_optic_inf = MockOptic(
            fields=[(0, -0.1), (0, 0.1)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        mock_optic_inf.update_paraxial_model()
        solver = RealFieldSolver()

        with self.assertRaisesRegex(RuntimeError, "could not find a solution"):
            solver.solve(mock_optic_inf, 20.0, tol=1e-7)

    def test_solve_real_optic_with_only_on_axis_field(self):
        # Optic only has [0,0] field defined.
        mock_optic_on_axis = MockOptic(
            fields=[(0, 0.0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        mock_optic_on_axis.update_paraxial_model()
        solver = RealFieldSolver()

        # Target is 0, should solve to 0.
        solved_field = solver.solve(mock_optic_on_axis, 0.0, tol=1e-7)
        self.assertAlmostEqual(solved_field, 0.0, places=6)

        # Target non-zero, but max field is 0. Should fail.
        # RealFieldSolver check: `if abs_max_y_field==0 and target_image_height!=0:`
        with self.assertRaisesRegex(
            ValueError, "Max field is 0, but target image height is non-zero"
        ):
            solver.solve(mock_optic_on_axis, 10.0, tol=1e-7)

    def test_solve_real_ray_trace_fails_at_boundaries(self):
        # Test scenario: ray tracing might fail at initial bracket boundaries.
        class MockOpticTraceFail(MockOptic):
            def trace_generic(self, ray_obj):
                Hy = self._current_Hy_for_real_trace
                # Fails if abs(Hy) > 0.5 rad
                if abs(Hy) > 0.5:
                    return MockRay(success=False)
                # Otherwise, behaves as perfect lens
                if self.object_is_at_infinity:
                    y_image = (self.efl_for_real_trace or 100.0) * Hy
                else:
                    y_image = (self.mag_for_real_trace or -1.0) * Hy
                return MockRay(y_at_image_surface=y_image, success=True)

        # Fields define wide range [-1.0, 1.0], but tracing fails beyond +/-0.5.
        mock_optic_fail = MockOpticTraceFail(
            fields=[(0, -1.0), (0, 1.0)], object_is_at_infinity=True, paraxial_efl=100.0
        )
        mock_optic_fail.update_paraxial_model()
        solver = RealFieldSolver()

        # Target img height 25.0. Expected field 0.25 (within traceable +/-0.5).
        # Initial bracket [-1.0, 1.0]. obj_func(-1.0), obj_func(1.0) -> np.inf.
        # Solver's bracketing logic tries sub-brackets like [0, 1.0] or [-1.0, 0].
        # If obj_func(0) * obj_func(0.5) has sign change, it should work.
        # obj_func(0) = 0 - 25 = -25. obj_func(0.5) = (100*0.5) - 25 = 25. Sign change.
        # So, bracket [0, 0.5] should be found and used.

        solved_field = solver.solve(mock_optic_fail, 25.0, tol=1e-7)
        self.assertAlmostEqual(solved_field, 0.25, places=6)

        # Target img height 75.0. Expected field 0.75 (outside traceable +/-0.5).
        # obj_func(0)=-75. obj_func(0.5)=(100*0.5)-75=-25. No sign change [0,0.5].
        # obj_func(-0.5)=(100*-0.5)-75=-125.
        # All finite obj_func values might have same sign if target large enough.
        with self.assertRaisesRegex(RuntimeError, "could not find a solution"):
            solver.solve(mock_optic_fail, 75.0, tol=1e-7)


if __name__ == "__main__":
    unittest.main()
