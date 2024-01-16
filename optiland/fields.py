import numpy as np


class Field:

    def __init__(self, field_type, x=0, y=0):
        self.field_type = field_type
        self.x = x
        self.y = y


class FieldGroup:

    def __init__(self):
        self.fields = []

    @property
    def x_fields(self):
        return np.array([field.x for field in self.fields])

    @property
    def y_fields(self):
        return np.array([field.y for field in self.fields])

    @property
    def max_x_field(self):
        return np.max(self.x_fields)

    @property
    def max_y_field(self):
        return np.max(self.y_fields)

    @property
    def max_field(self):
        return np.max(np.sqrt(self.x_fields**2 + self.y_fields**2))

    @property
    def num_fields(self):
        return len(self.fields)

    def get_field_coords(self):
        max_field = self.max_field
        if max_field == 0:
            return [(0, 0)]
        return [(x/max_field, y/max_field)
                for x, y in zip(self.x_fields, self.y_fields)]

    def add_field(self, field):
        self.fields.append(field)

    def get_field(self, field_number):
        return self.fields[field_number]
