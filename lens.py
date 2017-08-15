# -*- coding: utf-8 -*-
"""
# lens.py
"""

from surface import surface
import numpy as np
from paraxial import paraxial

class lens(paraxial):
    
    def __init__(self, name=''):
        super().__init__(self)
        
        self.name = name
        self.num_surfaces = 0
        self.wavelengths_list = []
        self.aperture_type = 'EPD'
        self.aperture_value = 1
        self.field_type = 'angle'
        self.fields_list = []
        self.stop = None
        self.prim_wave = 0
        self.image_solve = False
        self.surface_list = []
        
        self.length_unit = 'mm'
        self.rot_unit = 'deg'
        
# *********************************************************************************
# ------ Lens properties ----------------------------------------------------------
# *********************************************************************************
    
    @property
    def R(self):
        self._R = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            self._R[k] = surf.radius
        return self._R
    
    @property
    def t(self):
        self._t = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            self._t[k] = surf.thickness
        return self._t
        
    @property
    def v(self):
        self._v = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            self._v[k] = surf.abbe
        return self._v
    
    @property
    def SA(self):
        self._SA = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            self._SA[k] = surf.semi_aperture
        return self._SA
        
    @property
    def C(self):
        return 1/self.R
        
    @property
    def surf_type(self):
        self._surf_type = []
        for surf in self.surface_list:
            self._surf_type.append(surf.type)
        return self._surf_type
        
    @property
    def N(self):
        return len(self.C)
        
    @property
    def decenter(self):
        dx, dy = np.zeros(self.num_surfaces), np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            dx[k] = surf.decenter[0]
            dy[k] = surf.decenter[1]
        return dx, dy
        
    @property
    def tilt(self):
        rx = np.zeros(self.num_surfaces)
        ry = np.zeros(self.num_surfaces)
        rz = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            rx[k] = surf.tilt[0]
            ry[k] = surf.tilt[1]
            rz[k] = surf.tilt[2]
        return rx, ry, rz
        
    def n(self, wave):
        self._n = np.zeros(self.num_surfaces)
        for k,surf in enumerate(self.surface_list):
            self._n[k] = surf.index[wave] 
        return self._n
        
# *********************************************************************************
# ------ Setting functions --------------------------------------------------------
# *********************************************************************************
        
    def set_R(self, value, surface):
        self.surface_list[surface].radius = value
        return self
        
    def set_t(self, value, surface):
        self.surface_list[surface].thickness = value
        return self
        
    def set_n(self, value, surface, wave):
        self.surface_list[surface].index[wave] = value
        return self
        
# *********************************************************************************
# ------ Surface functions --------------------------------------------------------
# *********************************************************************************
    def add_surface(self, number, type_='standard', radius=1e10, thickness=0,\
                 conic=0, material='air', coating=None, semi_aperture=5, comment='',\
                 stop=False, *args, **kwargs):
                     
        if stop:
            if self.stop is None:
                self.stop = number
            else:
                raise ValueError('Redefinition of stop position at surface %d'%number)
                     
        self.num_surfaces += 1
        self.surface_list.append(surface(number, self.wavelengths_list, type_=type_,\
                                         radius=radius, thickness=thickness, conic=conic, material=material,\
                                         coating=coating, semi_aperture=semi_aperture, comment=comment,\
                                         stop=stop, *args, **kwargs))
                                         
    def list_surfaces(self):
        for surf in self.surface_list:
            print('---------------- Surface %d ----------------'%surf.number)
            surf.info()
            print('--------------------------------------------\n')
        return self
            
    def stop_surf(self):
        for surf in self.surface_list:
            if surf.stop:
                return surf.number
        return None
        
    def update_index_lists(self):
        for surf in self.surface_list:
            surf.update_index()
            
    def update_abbe_list(self):
        for surf in self.surface_list:
            surf.update_abbe()

# *********************************************************************************
# ------ Wavelength functions -----------------------------------------------------
# *********************************************************************************                 
    def add_wavelength(self, number, value, primary=False):
        
        if primary:
            self.prim_wave = number
        
        self.wavelengths_list.insert(number, value)
        self.update_index_lists()
        self.update_abbe_list()
        return self
        
    def list_wavelengths(self):
        for k, wavelength in enumerate(self.wavelengths_list):
            print('Wavelength %d: %f'%(k,wavelength))
        return self

# *********************************************************************************
# ------ Field functions ----------------------------------------------------------
# *********************************************************************************     
    def add_field(self, number, x, y):
        self.fields_list.insert(number,[x,y])
        return self
        
    def list_fields(self):
        print('Field Type: %s'%self.field_type)
        for k, field in enumerate(self.fields_list):
            print('Field ' + str(k) + ': ' + str(field))
        return self
        
    def x_field_list(self):
        return [x[0] for x in self.fields_list]
        
    def y_field_list(self):
        return [x[1] for x in self.fields_list]
        
# *********************************************************************************
# ------ Aperture functions ----------------------------------------------------------
# ********************************************************************************* 
        
    def list_aperture(self):
        print('Aperture Type: %s'%self.aperture_type)
        print('Aperture Value: ' + str(self.aperture_value))
        return self
                
if __name__ == '__main__':
    
    # create lens
    singlet = lens(name='Singlet')

    # create surfaces    
    singlet.add_surface(number=0,thickness=1e10,comment='object')
    singlet.add_surface(number=1,radius=26.2467,material='SF6',thickness=5,stop=True)
    singlet.add_surface(number=2,thickness=28.501)
    singlet.add_surface(number=3,comment='image')
    
    # add fields
    singlet.field_type = 'angle'
    singlet.add_field(number=0,x=0,y=0)
    singlet.add_field(number=1,x=0,y=7)
    singlet.add_field(number=2,x=0,y=10)
    
    # add aeperture type and value
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    singlet.add_wavelength(number=0, value=450)
    singlet.add_wavelength(number=1, value=550, primary=True)
    singlet.add_wavelength(number=2, value=650)
    
    singlet.list_surfaces()
    singlet.list_fields()
    singlet.list_wavelengths()
    singlet.list_aperture()
    
    
    