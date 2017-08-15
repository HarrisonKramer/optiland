# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 16:51:03 2017

@author: kramer
"""

class ray(object):
    '''
    ray class
    
    This class defines a light ray.
    
    Attributes:
        x - x position of ray start
        y - y position of ray start
        z - z position of ray start
        K - x-axis direction cosine of ray
        L - y-axis direction cosine of ray
        M - z-axis direction cosine of ray
        OPL - optical path length of ray (default=None)
        rel_inten - (0-1) the relative intensity of the ray. This value is relative
                    to the peak power of the incidident light on a system.
    
    '''
    
    def __init__(self, x, y, z, K, L, M, OPL=None, rel_inten=1):
        self.x = x
        self.y = y
        self.z = z
        self.K = K
        self.L = L
        self.M = M
        self.OPL = OPL
        self.rel_inten = rel_inten
        
    def info(self):
        print('x: %.6f'%self.x)
        print('y: %.6f'%self.y)
        print('z: %.6f'%self.z)
        print('K: %.8f'%self.K)
        print('L: %.8f'%self.L)
        print('M: %.8f'%self.M)
        print('OPL: ' + str(self.OPL))