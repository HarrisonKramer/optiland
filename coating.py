# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:54:56 2017

@author: kramer
"""
import numpy as np

class layer(object):
    
    def __init__(self, material, thickness):
        self.material = material
        self.thickness = thickness # nm
    
    def char_mat(self, wavelength, vr, pol=None):
        '''characteristic matrix
        
        vr = angle of refraction (i.e. angle through layer)
        wavelength units = nm
        
        '''
        Nr = self.material.n(wavelength)
        if pol == 'p':
            eta = Nr/np.cos(vr)
        elif pol == 's':
            eta = Nr*np.cos(vr)
        elif pol is None:
            eta = Nr
        dr = 2*np.pi*Nr*self.thickness*np.cos(vr)/wavelength
        
        return np.array([[np.cos(dr),1j*np.sin(dr)/eta],[1j*eta*np.sin(dr),np.cos(dr)]])
        
            
    
class coating(object):
    
    def __init__(self, name=''):
        self.name = name
        self.layer_list = []
    
    def add_layer(self, material, thickness):
        self.layer_list.append(layer(material, thickness))
        
    def n(self, wavelength):
        n = []
        for layer in self.layer_list:
            n.append( layer.material.n(wavelength) )
        return np.array(n)
        
    def _vrs(self, wavelength, vr0, n0):
        '''find all AOIs (vrs) for all layers within coating stack'''
        return np.arcsin(n0*np.sin(vr0)/self.n(wavelength))
        
    def _char_mat(self, wavelength, vr0, n0, pol=None):
        vrs = self._vrs(wavelength, vr0, n0)
        mat = np.eye(2)
        for idv, layer in enumerate(self.layer_list):
            mat = mat.dot(layer.char_mat(wavelength, vrs[idv], pol))
        return mat
        
    def _eta(self, n, vr=0, pol=None):
        if pol == 'p':
            eta = n/np.cos(vr)
        elif pol == 's':
            eta = n*np.cos(vr)
        elif pol is None:
            eta = n
        return eta
        
    def _Y(self, sub_mat, wavelength, vr0, n0, pol=None):
        nm = sub_mat.n(wavelength)
        E = np.array([1,self._eta(nm)])
        return self._char_mat(wavelength, vr0, n0, pol).dot(E)
    
    def R(self, incident_mat, sub_mat, AOI, wavelength, pol=None):
        '''reflectance of the coating stack at AOI, wavelength, and pol'''
        n0 = incident_mat.n(wavelength)
        B,C = self._Y(sub_mat, wavelength, AOI, n0, pol)
        eta0 = self._eta(n0, AOI, pol)
        print(B,C,n0,eta0)
        return np.real(np.conj((eta0-C/B)/(eta0+C/B))*((eta0-C/B)/(eta0+C/B)))
    
    def T(self, incident_mat, sub_mat, AOI, wavelength, pol=None):
        '''transmittance of the coating stack at AOI, wavelength, and pol'''
        pass
    

def plotR(coating, incident_mat, sub_mat, AOI, wavemin, wavemax, pol=None):
    
    pass

    
if __name__ == '__main__':
    
    from material import material
    
    x = coating()
    name = 'SF6'
    wavelengths = [450,550,650]
    g = material(name,wavelengths)
    x.add_layer(g,137.5)
    
    incident_mat = material('BAC4',wavelengths)
    sub_mat = material('N-BK7',wavelengths)
    
    r = x.R(incident_mat,sub_mat,0,wavelengths[1])
    print(r)
    
    
    
    
    
    
    
    
    
    
    