# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:09:30 2017

@author: kramer
"""

import yaml
import numpy as np
import glob

class glass(object):
    
    def __init__(self, glassname, wavelengths, manufacturer=None):
        self.glassname = glassname
        self.wavelengths = wavelengths
        self.manufacturer = manufacturer
        
        self._retrieve_file()
        self._read_yaml()
        self._decipher_type()
        self._get_coeffs()
        
    def _retrieve_file(self):
        self.files = []
        
        if self.manufacturer is None:
            reg_path = 'database/**/%s.yml'%self.glassname
        else:
            reg_path = 'database/**/%s/%s.yml'%(self.manufacturer,self.glassname)
        
        for filename in glob.iglob(reg_path, recursive=True):
            self.files.append(filename)
            
        if not self.files:
            raise ValueError('No glass data found for glassname %s'%self.glassname)
        if len(self.files) > 1:
            error_str = 'More than one glass manufacturer found for glassname ' +\
                        str(self.glassname) + ': ' + str(self.files) + '. Please additionally list manufacturer' +\
                        ' when adding new surface.'
            raise ValueError(error_str)
        
    def _read_yaml(self):
        with open(self.files[0], 'r') as stream:
            self.data = yaml.load(stream)
        
    def _decipher_type(self):
        self.types = []
        for each in self.data['DATA']:
            if each['type'] is not None:
                self.types.append(each['type'])
    
    def _get_coeffs(self):
        self.coeffs = [float(k) for k in self.data['DATA'][0]['coefficients'].split()]
        
    def _n(self, wavelength):
        L = wavelength/1000
        C = self.coeffs
        for formula in self.types:
            if formula == 'formula 1':
                n = 1 + C[0]
                for k in range(8):
                    n += C[2*k]*L**2/(L**2 - C[2*k+1]**2)
                return np.sqrt(n)
                
            elif formula == 'formula 2':
                n = np.sqrt(1 + C[0] + C[1]*L**2/(L**2 - C[2]) + C[3]*L**2/(L**2 - C[4]) + C[5]*L**2/(L**2 - C[6]))
                return n
                
            elif formula == 'formula 3':
                n = C[0]
                for k in range(1,len(C),2):
                    n += C[k]*L**C[k+1]
                return np.sqrt(n)
            
            else:
                return None
                
    def index(self):
        
        n = []
        for wave in self.wavelengths:
            n.append(self._n(wave))
        return n
        
    def abbe(self):
        nD = self._n(589.3)
        nF = self._n(486.1)
        nC = self._n(656.3)
        return (nD-1)/(nF-nC)

if __name__ == '__main__':
    
    glassname = 'SF6'
    wavelengths = [450,550,650]
    g = glass(glassname,wavelengths)
    print(g.index())
    print(g.abbe())
    



    
    

        
    