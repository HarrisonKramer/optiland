# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 17:01:00 2017

@author: kramer
"""

import numpy as np

from ray import ray

class transform(object):
    '''class to transform a coordinate system during ray tracing, e.g. for tilted 
       or decentered surfaces'''
    
    def __init__(self, ray, tilt=[0,0,0], decenter=[0,0]):
        '''
        Inputs:
            tilt - 3-element list of tilts in degrees around axes x, y, and z, respectively.
            decenter - 2-element list of decenters (in lens units) in axes x and y, respectively.
        '''
        self.ray = ray
        self.tilt = np.deg2rad(tilt)
        self.decenter = decenter
        
    def Rx(self, sign=1):
        return np.array([[1,0,0],
                         [0,np.cos(self.tilt[0]),-np.sin(sign*self.tilt[0])],
                         [0,np.sin(sign*self.tilt[0]),np.cos(self.tilt[0])]])
    
    def Ry(self, sign=1):
        return np.array([[np.cos(self.tilt[1]),0,np.sin(sign*self.tilt[1])],
                         [0,1,0],
                         [-np.sin(sign*self.tilt[1]),0,np.cos(self.tilt[1])]])
    
    def Rz(self, sign=1):
        return np.array([[np.cos(self.tilt[2]),-np.sin(sign*self.tilt[2]),0],
                         [np.sin(sign*self.tilt[2]),np.cos(self.tilt[2]),0],
                         [0,0,1]])
    
    def R(self, sign=1):
        return self.Rz(sign)*self.Ry(sign)*self.Rz(sign)
    
    def run(self):
        dx,dy = self.decenter
        x = self.ray.x - dx
        y = self.ray.y - dy
        z = self.ray.z
        R = self.R()
        K,L,M = R.dot(np.array([self.ray.K,self.ray.L,self.ray.M]))
        return ray(x,y,z,K,L,M)
    
    def unrun(self):
        dx,dy = self.decenter
        x = self.ray.x + dx
        y = self.ray.y + dy
        z = self.ray.z
        R = self.R(sign=-1)
        K,L,M = R.dot(np.array([self.ray.K,self.ray.L,self.ray.M]))
        return ray(x,y,z,K,L,M)
        
        