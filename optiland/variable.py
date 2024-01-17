class Variable:

    def __init__(self, optic, type, **kwargs):
        self.__dict__.update(kwargs)
        self.optic = optic
        self.type = type

        self._surfaces = self.optic.surface_group

        if not hasattr(self, 'min_val'):
            self.min_val = None

        if not hasattr(self, 'max_val'):
            self.max_val = None

    @property
    def value(self):
        if self.type == 'radius':
            return self._surfaces.radii[self.surface_number]
        elif self.type == 'conic':
            return self._surfaces.conic[self.surface_number]
        elif self.type == 'thickness':
            return self._surfaces.get_thickness(self.surface_number)[0]
        elif self.type == 'index':
            n = self.optic.n(self.wavelength)
            return n[self.surface_number]
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')

    @property
    def bounds(self):
        '''return the bounds of the variable'''
        return (self.min_val, self.max_val)

    def update(self, new_value):
        '''update variable to a new value'''
        if self.type == 'radius':
            self.optic.set_radius(new_value, self.surface_number)
        elif self.type == 'conic':
            self.optic.set_conic(new_value, self.surface_number)
        elif self.type == 'thickness':
            self.optic.set_thickness(new_value, self.surface_number)
        elif self.type == 'index':
            self.optic.set_thickness(new_value, self.surface_number)
        else:
            raise ValueError(f'Invalid variable type "{self.type}"')
