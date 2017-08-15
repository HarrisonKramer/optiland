# -*- coding: utf-8 -*-
        
class variable(object):
    
    def __init__(self, number, lens, type_, **kwargs):
        self.__dict__.update(kwargs)
        self.number = number
        self.lens = lens
        self.type_ = type_
        
        if not hasattr(self,'min_val'):
            self.min_val = None
            
        if not hasattr(self,'max_val'):
            self.max_val = None
    
    @property
    def value(self):
        if self.type_ == 'radius':
            return self.lens.surface_list[self.surf_num].radius
        elif self.type_ == 'thickness':
            return self.lens.surface_list[self.surf_num].thickness
        elif self.type_ == 'index':
            return self.lens.surface_list[self.surf_num].index[self.lens.prim_wave]
        elif self.type_ == 'abbe':
            return self.lens.surface_list[self.surf_num].abbe
        elif self.type_ == 'aspheric_coeff':
            return self.lens.surface_list.A[self.coeff]
        else:
            raise ValueError('Invalid variable type ' + str(self.type_))
    
    @property
    def bounds(self):
        '''return the bounds of the variable'''
        return (self.min_val, self.max_val)
        
    def info(self):
        print('\t Number: %d'%self.number)
        print('\t   Type: %s'%self.type_)
        print('\t   Value %.6f'%self.value)
        
    def update(self, new_value):
        '''update variable to a new value'''
        if self.type_ == 'radius':
            self.lens.surface_list[self.surf_num].radius = new_value
        elif self.type_ == 'thickness':
            self.lens.surface_list[self.surf_num].thickness = new_value
        elif self.type_ == 'index':
            self.lens.surface_list[self.surf_num].index =  [new_value for x in self.lens.wavelengths_list]
        elif self.type_ == 'abbe':
            self.lens.surface_list[self.surf_num].abbe = new_value
        elif self.type_ == 'aspheric_coeff':
            self.lens.surface_list.A[self.coeff] = new_value
        else:
            raise ValueError('Invalid variable type ' + str(self.type_))
        return self
















