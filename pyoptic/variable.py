# TODO: update using Variable classes, like operand.py
class Variable:

    def __init__(self, optic, variable_type, **kwargs):
        self.__dict__.update(kwargs)
        self.optic = optic
        self.variable_type = variable_type

        self._surfaces = self.optic.surface_group

        if not hasattr(self, 'min_val'):
            self.min_val = None

        if not hasattr(self, 'max_val'):
            self.max_val = None

    @property
    def value(self):
        if self.variable_type == 'radius':
            return self._surfaces.radii[self.surface_number]
        elif self.variable_type == 'thickness':
            return self._surfaces.get_thickness(self.surface_number)
        elif self.variable_type == 'index':
            n = self.optic.n(self.wavelength)
            return n[self.surface_number]
        else:
            raise ValueError(f'Invalid variable type "{self.variable_type}"')

    @property
    def bounds(self):
        '''return the bounds of the variable'''
        return (self.min_val, self.max_val)

    def info(self):
        print(f'\n\t   Type: {self.variable_type}')
        print(f'\t   Value: {self.value}')

    def update(self, new_value):
        '''update variable to a new value'''
        if self.variable_type == 'radius':
            self.optic.set_radius(new_value, self.surface_number)
        elif self.variable_type == 'thickness':
            self.optic.set_thickness(new_value, self.surface_number)
        elif self.variable_type == 'index':
            self.optic.set_thickness(new_value, self.surface_number)
        else:
            raise ValueError(f'Invalid variable type "{self.variable_type}"')
