# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:48:57 2017

@author: kramer
"""

from ray import ray
import numpy as np

class refract(object):
    '''class for refracting at a surface'''
    
    def __init__(self, surf1, surf2, ray, wave, **kwargs):
        '''create instance of refract class
        
        Attributes:
            surf - surface at which to refract
            ray - instance of the ray class, which defines initial ray in coord.
                  system of surf vertex
        
        Methods:
            _standard - transfer to a standard (spherical surface)
            _aspheric - transfer to an aspheric surface
            run - perform transfer between surf1 and surf2
            
        Usage:
            ray_out = refract(ray_in)
        '''
        self.__dict__.update(kwargs)
        self.surf1 = surf1
        self.surf2 = surf2
        self.ray = ray
        self.wave = wave
        
    def _standard(self):
        '''refract at spherical surface "surf"'''
        C = 1/self.surf2.radius
        n0 = self.surf1.index[self.wave]
        n1 = self.surf2.index[self.wave]
        e = -self.ray.x*self.ray.K - self.ray.y*self.ray.L - self.ray.z*self.ray.M
        Mz = self.ray.z + e*self.ray.M
        M2 = self.ray.x**2 + self.ray.y**2 + self.ray.z**2 - e**2
        arg = self.ray.M**2 - C*(C*M2 - 2*Mz)
        if arg < 0:
            print('Warning: Total internal reflection occurs on surface %d'%self.surf2.number)
        E1 = np.sqrt(arg)
        Ep = np.sqrt(1 - (n0/n1)**2*(1 - E1**2))
        g = Ep - n0/n1*E1
        K1 = n0/n1*self.ray.K - g*C*self.ray.x
        L1 = n0/n1*self.ray.L - g*C*self.ray.y
        M1 = n0/n1*self.ray.M - g*C*self.ray.z + g
        return ray(self.ray.x,self.ray.y,self.ray.z,K1,L1,M1)
    
    def _asphere(self):
        '''refract at an aspheric surface'''
        C = 1/self.surf2.radius
        n0 = self.surf1.index[self.wave]
        n1 = self.surf2.index[self.wave]
        r2 = self.ray.x**2 + self.ray.y**2
        m0 = np.sqrt(1 - C**2*r2)
        arg = C + m0*sum([2*(k+1)*r2**k*a for k,a in enumerate(self.surf2.A)])
        l0 = -self.ray.y*arg
        k0 = -self.ray.x*arg
        P2 = k0**2 + l0**2 + m0**2
        F = self.ray.K*k0 + self.ray.L*l0 + self.ray.M*m0
        arg = P2*(1-n0**2/n1**2) + n0**2/n1**2*F**2
        if arg < 0:
            print('Warning: Total internal reflection occurs on surface %d'%self.surf2.number)
        Fp = np.sqrt(arg)
        g = 1/P2*(Fp - n0/n1*F)
        K1 = n0/n1*self.ray.K + g*k0
        L1 = n0/n1*self.ray.L + g*l0
        M1 = n0/n1*self.ray.M + g*m0
        return ray(self.ray.x,self.ray.y,self.ray.z,K1,L1,M1)
    
    def run(self):
        '''perform refraction'''
        if self.surf2.type == 'standard':
            return self._standard()
        elif self.surf2.type == 'aspheric':
            return self._asphere()
        else:
            raise ValueError('Invalid surface type specified.')