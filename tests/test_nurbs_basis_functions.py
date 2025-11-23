"""Unit tests for the nurbs_basis_functions module."""
import pytest
from optiland.geometries.nurbs import nurbs_basis_functions

def test_basis_function():
    """Tests the basis_function."""
    degree = 2
    knot_vector = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    span = 2
    knot = 0.5

    result = nurbs_basis_functions.basis_function(degree, knot_vector, span, knot)

    assert isinstance(result, list)
    assert len(result) == degree + 1
    assert all(isinstance(x, float) for x in result)

def test_basis_function_one():
    """Tests the basis_function_one."""
    degree = 2
    knot_vector = [0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0]

    # Test a knot inside the span
    result_inside = nurbs_basis_functions.basis_function_one(degree, knot_vector, 2, 0.25)
    assert isinstance(result_inside, float)
    assert 0.0 < result_inside < 1.0

    # Test edge case: knot at the start of the vector
    result_start = nurbs_basis_functions.basis_function_one(degree, knot_vector, 0, 0.0)
    assert result_start == 1.0

    # Test edge case: knot at the end of the vector
    result_end = nurbs_basis_functions.basis_function_one(degree, knot_vector, 3, 1.0)
    assert result_end == 1.0

    # Test knot outside the valid range
    result_outside = nurbs_basis_functions.basis_function_one(degree, knot_vector, 2, 1.1)
    assert result_outside == 0.0
