# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:49:06 2017

@author: kramer
"""

from ray import ray
import numpy as np

class transfer(object):
    '''class for transfering from one surface to another'''
    
    def __init__(self, surf1, surf2, ray, wave, **kwargs):
        '''create instance of transfer class
        
        Attributes:
            surf1 - surface from which to transfer
            surf2 - surface to which to transfer
            ray - instance of the ray class, which defines initial ray in coord.
                  system of surf1 vertex
        
        Methods:
            _standard - transfer to a standard (spherical surface)
            _aspheric - transfer to an aspheric surface
            run - perform transfer between surf1 and surf2
            
        Usage:
            ray_out = transfer(surf1, surf2, ray_in, wave)
        
        '''
        self.__dict__.update(kwargs)
        self.surf1 = surf1
        self.surf2 = surf2
        self.ray = ray
        self.wave = wave
    
    # *************************************************************************************************
    def _standard(self, EP_transfer=False, **kwargs):
        '''transfer from point x,y,z with dir. cosines K,L,M to spherical surface "surf2"'''
        if EP_transfer:
            t = kwargs['t']
        else:
            t = self.surf1.thickness
        C = 1/self.surf2.radius
        e = t*self.ray.M - (self.ray.x*self.ray.K + self.ray.y*self.ray.L + self.ray.z*self.ray.M)
        Mz = self.ray.z + e*self.ray.M - t
        M2 = self.ray.x**2 + self.ray.y**2 + self.ray.z**2 - e**2 + t**2 - 2*t*self.ray.z
        arg = self.ray.M**2 - C*(C*M2 - 2*Mz)
        if arg < 0:
            print('Warning: Ray does not intersect surface %d'%self.surf2.number)
        E1 = np.sqrt(arg)
        OPL = e + (C*M2 - 2*Mz)/(self.ray.M + E1)
        x1 = self.ray.x + OPL*self.ray.K
        y1 = self.ray.y + OPL*self.ray.L
        z1 = self.ray.z + OPL*self.ray.M - t
        
        OPL = OPL*self.surf1.index[self.wave]
        return ray(x1,y1,z1,self.ray.K,self.ray.L,self.ray.M,OPL=OPL)
    # *************************************************************************************************
    def _sag_aspheric(self, r2):
        '''calculates the sag (z) of surf2 at point x,y'''
        C = 1/self.surf2.radius
        k = self.surf2.conic
        z = C*r2/(1 + np.sqrt(1 - C**2*r2*(k+1))) + sum([r2**(m+1)*a for m,a in enumerate(self.surf2.A)])      
        return z
        
    def _aspheric(self, EP_transfer=False, **kwargs):
        '''transfer from point x,y,z with dir. cosines K,L,M to aspheric surface "surf2"'''
        if EP_transfer:
            t = kwargs['t']
            ray = self._standard(EP_transfer=True, t=t) # intersection of ray with the spherical surface
        else:
            ray = self._standard() # intersection of ray with the spherical surface
        C = 1/self.surf2.radius
        zb = 1e10
        while np.abs(zb-ray.z) > 1e-6:
            r2 = ray.x**2+ray.y**2
            zb = self._sag_aspheric(r2)
            m0 = np.sqrt(1 - C**2*r2)
            arg = C + m0*sum([2*(k+1)*r2**k*a for k,a in enumerate(self.surf2.A)])
            l0 = -ray.y*arg
            k0 = -ray.x*arg
            G = m0*(zb - ray.z)/(ray.K*k0 + ray.L*l0 + ray.M*m0)
            
            ray.x = G*ray.K + ray.x
            ray.y = G*ray.L + ray.y
            ray.z = G*ray.M + ray.z
            ray.OPL += G*self.surf1.index[self.wave]
            
        return ray
        
        
    # *************************************************************************************************
    def run(self, EP_transfer=False, **kwargs):
        ''''perform transfer''' 
        if self.surf2.type == 'standard':
            return self._standard(EP_transfer=EP_transfer, **kwargs)
        elif self.surf2.type == 'aspheric':
            return self._aspheric(EP_transfer=EP_transfer, **kwargs)
        else:
            raise ValueError('Invalid surface type specified.')