# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:18:00 2017

@author: kramer
"""

from trace import trace
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class wavefront(object):
    
    def __init__(self, lens, distribution, Hx=0, Hy=0, wave=0):
        self.lens = lens
        self.distribution = distribution
        self.Hx = Hx
        self.Hy = Hy
        self.wave = wave
        
    def _chief_ray(self):
        '''return final x/y/z/K/L/M of chief ray on image surface'''
        raytrace = trace(self.lens,self.Hx,self.Hy,0,0,self.wave).run()
        return raytrace.ray_list[-1]
        
    def _chief_intersection(self):
        '''intersection point of chief ray with image plane in x,y,z'''
        r = self._chief_ray()
        return np.array([r.x,r.y,r.z])
    
    # TODO - verify that using paraxial XP location is valid
    def _center_ref_sphere(self):
        '''center of the reference sphere w.r.t. image plane'''
        return np.array([0,0,self.lens.XPL()-self.lens.t[-2]])
        
    def _radius_ref_sphere(self):
        '''radius of the reference sphere'''
        P0 = self._chief_intersection()
        C = self._center_ref_sphere()
        return np.linalg.norm(P0-C)
        
    def _ray_direction(self, ray):
        '''direction vector of a given ray'''
        return np.array([ray.K,ray.L,ray.M])
        
    def _ray_intersection(self, ray):
        '''intersection of a ray with its respective surface'''
        return np.array([ray.x,ray.y,ray.z])
        
    def _OPL_XPtoImg(self, ray, C, R):
        '''OPL from exit pupil to image plane for a given ray'''
        L = self._ray_direction(ray)
        O = self._ray_intersection(ray)
        n = self.lens.n(self.lens.prim_wave)[-1] # image space index for primary wavelength
        root = L.dot(O-C)**2 - np.linalg.norm(O-C)**2 + R**2
        if root < 0:
            print('Warning: Ray does not intersect reference sphere')
        else:
            return n*(2*L.dot(O-C) - np.sqrt(root))
            
    def _ray_OPL(self,Px,Py,C,R):
        '''OPL from object to reference sphere'''
        raytrace = trace(self.lens,self.Hx,self.Hy,Px,Py,self.wave).run()
        return sum(raytrace.OPL) - self._OPL_XPtoImg(raytrace.ray_list[-1], C, R)
        
    def OPD_array(self):
        '''find OPL array of wavefront at exit pupil'''
        C = self._center_ref_sphere()
        R = self._radius_ref_sphere()
        OPL_chief = self._ray_OPL(0,0,C,R)
        OPD = []
        for Px,Py in self.distribution.points:
            OPD.append(self._ray_OPL(Px,Py,C,R) - OPL_chief)
        return OPD
        
    def phase_array(self):
        '''phase delay array over exit pupil'''
        n = self.lens.n(self.lens.prim_wave)[-1] # image space index for primary wavelength
        L = self.lens.wavelengths_list[self.lens.prim_wave]
        OPD = self.OPD_array()
        phase = [2*np.pi*n/L*x for x in OPD]
        return phase
        
    def show_OPD(self):
        '''show the OPD array'''
        
        OPD = self.OPD_array()
        x,y = zip(*self.distribution.points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,OPD)
        ax.set_title('Wavefront OPD Map')
        ax.set_xlabel('Pupil X Coordinate')
        ax.set_ylabel('Pupil Y Coordinate')
        ax.set_zlabel('Optical Path Difference (lens units)')
        plt.show()
        
    def show_phase(self):
        '''show the phase OPD array'''
        
        phase = self.phase_array()
        x,y = zip(*self.distribution.points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,phase)
        ax.set_title('Wavefront OPD Phase Map')
        ax.set_xlabel('Pupil X Coordinate')
        ax.set_ylabel('Pupil Y Coordinate')
        ax.set_zlabel('Optical Path Difference, Phase (radians)')
        plt.show()

if __name__ == '__main__':
    
    from distribution import distribution
    import sample_lenses
        
    system = sample_lenses.double_Gauss()
    pupil_grid = distribution(type_='rectangular')

    wvfrnt = wavefront(system,pupil_grid,Hx=0,Hy=1)
    wvfrnt.show_phase()
    
    