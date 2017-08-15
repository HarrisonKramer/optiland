# -*- coding: utf-8 -*-
"""
# surface.py
"""

from material import material

class surface(object):
    
    def __init__(self, number, wavelengths, type_='standard', radius=1e10, thickness=0,\
                 stop=False, conic=0, material='air', coating=None, semi_aperture=5, comment='',\
                 decenter=[0,0], tilt=[0,0,0], *args, **kwargs):
                     
        for key, val in kwargs.items():
            setattr(self, key, val)
        self.number = number
        self.wavelengths = wavelengths
        self.type = type_
        self.radius = radius
        self.thickness = thickness
        self.stop = stop
        self.conic = conic
        self.material = material
        self.update_index()
        self.update_abbe()
        self.coating = coating
        self.semi_aperture = semi_aperture
        self.comment = comment
        self.decenter = decenter
        self.tilt = tilt
        
    @property
    def semi_aperture(self):
        return self._semi_aperture
    
    @semi_aperture.setter
    def semi_aperture(self, value):
        if self.number == 0:
            self._semi_aperture = None
        else:
            self._semi_aperture = value
            
    @property
    def radius(self):
        return self._radius
        
    @radius.setter
    def radius(self, value):
        if self.number == 0:
            self._radius = 1e10
        else:
            self._radius = value
            
    @property
    def conic(self):
        return self._conic
        
    @conic.setter
    def conic(self, value):
        if value != 0:
            self.type = 'aspheric'
            if not hasattr(self,'A'):
                self.A = [0]
        self._conic = value
        
    @property
    def decentered(self):
        if self.decenter == [0,0]:
            return False
        else:
            return True
            
    @property
    def tilted(self):
        if self.tilt == [0,0,0]:
            return False
        else:
            return True
            
    def update_glass(self):
        if hasattr(self, 'manufacturer'):
            self.glass = material(self.material, self.wavelengths, self.manufacturer)
        else:
            self.glass = material(self.material, self.wavelengths)
            
    def update_index(self):
        if self.material.lower() == 'air':
            self.index = [1.0] * len(self.wavelengths)
        elif self.material.lower() == 'mirror':
            self.index = [-1.0] * len(self.wavelengths)
        else:
            self.update_glass()
            self.index = self.glass.index()
            
    def update_abbe(self):
        if self.material.lower() in ['air','mirror']:
            self.abbe = 1.0
        else:
            self.update_glass()
            self.abbe = self.glass.abbe()
        
    def info(self):
        print('Surface Number: ' + str(self.number))
        print('Wavelengths: ' + str(self.wavelengths))
        print('Type: ' + str(self.type))
        print('Thickness: ' + str(self.thickness))
        print('Stop: ' + str(self.stop))
        print('Conic: ' + str(self.conic))
        print('Index: ' + str(self.index))
        print('Abbe V Number: ' + str(self.abbe))
        print('Coating: ' + str(self.coating))
        print('Semi-Aperture: ' + str(self.semi_aperture))
        print('Comment: ' + str(self.comment))
        
if __name__ == '__main__':
    
    new_surface = surface(number=0, type_='standard', conic=-1,wavelengths=[400,500,600], radius=100, thickness=10, material='N-BK7')
    new_surface.info()