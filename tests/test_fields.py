import pytest
import numpy as np
from optiland import fields


@pytest.mark.parametrize('field_type, x, y',
                         [('angle', 0, 0),
                          ('object_height', 5.3, 8.5),
                          ('angle', 0, 4.2)])
def test_field(field_type, x, y):
    f = fields.Field(field_type, x, y)

    assert f.field_type == field_type
    assert f.x == x
    assert f.y == y


def test_field_group_inputs():
    input_data = [(0, 0), (5, 0), (0, 6), (7, 9.2)]
    f = fields.FieldGroup()
    for field_data in input_data:
        new_field = fields.Field('angle', *field_data)
        f.add_field(new_field)

    assert np.array_equal(f.x_fields, np.array([0, 5, 0, 7]))
    assert np.array_equal(f.y_fields, np.array([0, 0, 6, 9.2]))

    assert f.max_x_field == 7
    assert f.max_y_field == 9.2
    assert f.max_field == np.sqrt(7**2 + 9.2**2)


def test_field_group_getters():
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = fields.FieldGroup()
    for field_data in input_data:
        new_field = fields.Field('angle', *field_data)
        f.add_field(new_field)

    assert f.get_field_coords() == [(0, 0), (2.5/5, 0), (0, 2/5), (4/5, 3/5)]

    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3

    # test case when max field is zero
    f = fields.FieldGroup()
    new_field = fields.Field('angle', 0, 0)
    f.add_field(new_field)
    assert f.get_field_coords() == [(0, 0)]


def test_field_group_get_vig_factor():
    input_data = [(0, 0)]
    f = fields.FieldGroup()
    for field_data in input_data:
        new_field = fields.Field('angle', *field_data)
        f.add_field(new_field)

    vx, ny = f.get_vig_factor(1, 1)
    assert vx == 0.0
    assert vx == 0.0

    input_data = [(0, 0), (0, 7), (0, 10)]
    f = fields.FieldGroup()
    for field_data in input_data:
        new_field = fields.Field('angle', *field_data,
                                 vignette_factor_x=0.2, vignette_factor_y=0.2)
        f.add_field(new_field)

    vx, ny = f.get_vig_factor(0.5, 0.7)
    assert vx == 0.2
    assert vx == 0.2

    vx, ny = f.get_vig_factor(1, 1)
    assert vx == 0.2
    assert vx == 0.2


def test_field_group_telecentric():
    f = fields.FieldGroup()
    assert f.telecentric is False

    f.set_telecentric(True)
    assert f.telecentric is True


def test_field_to_dict():
    f = fields.Field('angle', 0, 0)
    assert f.to_dict() == {'field_type': 'angle', 'x': 0, 'y': 0,
                           'vx': 0, 'vy': 0}


def test_field_group_to_dict():
    input_data = [(0, 0), (2.5, 0), (0, 2), (4, 3)]
    f = fields.FieldGroup()
    for field_data in input_data:
        new_field = fields.Field('angle', *field_data)
        f.add_field(new_field)

    assert f.to_dict() == {'fields': [{'field_type': 'angle', 'x': 0, 'y': 0,
                                       'vx': 0, 'vy': 0},
                                      {'field_type': 'angle', 'x': 2.5, 'y': 0,
                                       'vx': 0, 'vy': 0},
                                      {'field_type': 'angle', 'x': 0, 'y': 2,
                                       'vx': 0, 'vy': 0},
                                      {'field_type': 'angle', 'x': 4, 'y': 3,
                                       'vx': 0, 'vy': 0}],
                           'telecentric': False}


def test_field_from_dict():
    f = fields.Field.from_dict({'field_type': 'angle', 'x': 0, 'y': 0,
                                'vx': 0, 'vy': 0})
    assert f.field_type == 'angle'
    assert f.x == 0
    assert f.y == 0
    assert f.vx == 0
    assert f.vy == 0


def test_field_group_from_dict():
    f = fields.FieldGroup.from_dict(
        {'fields': [
            {'field_type': 'angle', 'x': 0, 'y': 0, 'vx': 0, 'vy': 0},
            {'field_type': 'angle', 'x': 2.5, 'y': 0, 'vx': 0, 'vy': 0},
            {'field_type': 'angle', 'x': 0, 'y': 2, 'vx': 0, 'vy': 0},
            {'field_type': 'angle', 'x': 4, 'y': 3, 'vx': 0, 'vy': 0}],
            'telecentric': False}
            )
    assert f.get_field(0).x == 0
    assert f.get_field(0).y == 0
    assert f.get_field(3).x == 4
    assert f.get_field(3).y == 3
    assert f.telecentric is False
