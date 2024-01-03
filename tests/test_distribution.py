import pytest
from optiland import create_distribution


def test_line_x():
    d = create_distribution('line_x')
    d.generate_points(num_points=10)

    assert d.dx == 0.1
    assert d.dy == 0.0
