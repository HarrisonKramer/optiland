# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 22:17:46 2017

@author: kramer
"""

import numpy as np
from distribution import distribution
from wavefront import wavefront
from zernike import fit
import matplotlib.pyplot as plt

class psf_fft(object):
    
    def __init__(self, lens, grid_size=40, zernike_fit=None, OPD_array=None, 
                 pupil_distribution=None, Hx=0, Hy=0, wave=0):
                     
        self.lens = lens
        self.grid_size = grid_size
        
        if zernike_fit is not None:
            if pupil_distribution is not None:
                print('Warning: Pupil distribution provided to psf constructor is unused' )
            if OPD_array is not None:
                print('Warning: Wavefront provided to psf constructor is unused' )
            self.fit = zernike_fit
            
        elif OPD_array is not None:
            if pupil_distribution is not None:
                print('Warning: Pupil distribution provided to psf constructor is unused' )
            self.fit = fit(OPD_array)
            
        else:
            dist = distribution('rectangular')
            phase = wavefront(lens, dist, Hx=Hx, Hy=Hy, wave=wave)
            self.fit = fit(phase)
            
    def run(self):
#        D = self.lens.XPD()   # diameter of exit pupil
#        
#        x = np.linspace(-D,D,2*self.grid_size)
#        [X,Y] = np.meshgrid(x,x)
#        Xs = X[int(D/2):int(3*D/2),int(D/2):int(3*D/2)] # sub-grid, over which OPD is non-zero
#        Ys = Y[int(D/2):int(3*D/2),int(D/2):int(3*D/2)]
#        R = np.sqrt(Xs**2+Ys**2)
#        R /= np.max(R)
#        phi = np.arctan2(Ys,Xs)
#        Zs = self.fit.poly(R,phi)
#        Z = np.zeros_like(X) # full grid
#        Z[int(D/2):int(3*D/2),int(D/2):int(3*D/2)] = Zs
    
        x = np.linspace(-1,1,2*self.grid_size)
        [X,Y] = np.meshgrid(x,x)
        R = np.sqrt(X**2+Y**2)
        phi = np.arctan2(Y,X)
        
#        X = X[np.abs(R)<1]
#        Y = Y[np.abs(R)<1]
#        phi = phi[np.abs(R)<1]
#        R = R[np.abs(R)<1]
        
        Z = self.fit.poly(R,phi)
        Z[np.abs(R)>1] = np.nan
        
        w = np.exp(-1j*2*np.pi*Z)
        PSF = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(w)))**2
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,PSF)
        plt.axis('image')
        plt.show()
        

def gaussian_beamlet_propagation():
    '''http://opticalengineering.spiedigitallibrary.org/article.aspx?articleid=2211786&resultClick=1'''
    pass
        


if __name__ == '__main__':
    
    import sample_lenses
    
    system = sample_lenses.Edmund_49_847()
    
    pupil_grid = distribution(type_='rectangular')

    sys_wavefront = wavefront(system,pupil_grid,Hx=0,Hy=1)
    sys_wavefront.show_phase()
    
    zern_fit = fit(sys_wavefront,num_terms=36,remove_piston=False)
    zern_fit.show(projection='3d', num_pts=200)
    
    psf = psf_fft(system, zernike_fit=zern_fit)
    psf.run()
    
    
    
    
    
    
    
    
    
    
    
    
    