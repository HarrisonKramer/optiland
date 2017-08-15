# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 19:05:18 2017

@author: kramer
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class zernike(object):
    '''Class for representing zernike fringe polynomial'''
    
    terms = ['piston',
             'tilt x',
             'tilt y',
             'power',
             'astig x',
             'astig y',
             'coma x',
             'coma y',
             'primary spherical',
             'trefoil x',
             'trefoil y',
             'secondary astig x',
             'seconday astig y',
             'seconday coma x',
             'seconday coma y',
             'secondary spherical',
             'tetrafoil x',
             'tetrafoil y',
             'secondary trefoil x',
             'secondary trefoil y',
             'tertiary astig x',
             'tertiary astig y',
             'tertiary coma x',
             'tertiary coma y',
             'tertiary spherical',
             'pentafoil x',
             'pentafoil y',
             'secondary tetrafoil x',
             'secondary tetrafoil y',
             'tertiary trefoil x',
             'tertiary trefoil y',
             'quaternary astig x',
             'quaternary astig y',
             'quaternary coma x',
             'quaternary coma y',
             'quaternary spherical']
    
    def __init__(self, Z=[0 for k in range(36)]):
        self.Z = Z
        
    # ==== Term Retrieval ======================================================
    def piston(self):
        return self.Z[0]
        
    def tilt_x(self):
        return self.Z[1]
        
    def tilt_y(self):
        return self.Z[2]
        
    def tilt(self):
        return np.sqrt(self.Z[1]**2 + self.Z[2]**2)
        
    def power(self):
        return self.Z[3]
        
    def astig_x(self):
        return self.Z[4]
        
    def astig_y(self):
        return self.Z[5]
        
    def astig(self):
        return np.sqrt(self.Z[4]**2 + self.Z[5]**2)
        
    def coma_x(self):
        return self.Z[6]
        
    def coma_y(self):
        return self.Z[7]
        
    def coma(self):
        return np.sqrt(self.Z[6]**2 + self.Z[7]**2)
        
    def primary_spherical(self):
        return self.Z[8]
        
    def trefoil_x(self):
        return self.Z[9]
        
    def trefoil_y(self):
        return self.Z[10]
        
    def trefoil(self):
        return np.sqrt(self.Z[9]**2 + self.Z[10]**2)
        
    def secondary_astig_x(self):
        return self.Z[11]
        
    def secondary_astig_y(self):
        return self.Z[12]
        
    def secondary_astig(self):
        return np.sqrt(self.Z[11]**2 + self.Z[12]**2)
        
    def secondary_coma_x(self):
        return self.Z[13]
        
    def secondary_coma_y(self):
        return self.Z[14]
        
    def secondary_coma(self):
        return np.sqrt(self.Z[13]**2 + self.Z[14]**2)
        
    def secondary_spherical(self):
        return self.Z[15]
        
    def tetrafoil_x(self):
        return self.Z[16]
        
    def tetrafoil_y(self):
        return self.Z[17]
        
    def tetrafoil(self):
        return np.sqrt(self.Z[16]**2 + self.Z[17]**2)
        
    def secondary_trefoil_x(self):
        return self.Z[18]
        
    def secondary_trefoil_y(self):
        return self.Z[19]
        
    def secondary_trefoil(self):
        return np.sqrt(self.Z[18]**2 + self.Z[19]**2)
        
    def tertiary_astig_x(self):
        return self.Z[20]
        
    def tertiary_astig_y(self):
        return self.Z[21]
        
    def tertiary_astig(self):
        return np.sqrt(self.Z[20]**2 + self.Z[21]**2)
        
    def tertiary_coma_x(self):
        return self.Z[22]
        
    def tertiary_coma_y(self):
        return self.Z[23]
        
    def tertiary_coma(self):
        return np.sqrt(self.Z[22]**2 + self.Z[23]**2)
        
    def tertiary_spherical(self):
        return self.Z[24]
        
    def pentafoil_x(self):
        return self.Z[25]
        
    def pentafoil_y(self):
        return self.Z[26]
        
    def pentafoil(self):
        return np.sqrt(self.Z[25]**2 + self.Z[26]**2)
        
    def secondary_tetrafoil_x(self):
        return self.Z[27]
        
    def secondary_tetrafoil_y(self):
        return self.Z[28]
        
    def secondary_tetrafoil(self):
        return np.sqrt(self.Z[27]**2 + self.Z[28]**2)
        
    def tertiary_trefoil_x(self):
        return self.Z[29]
        
    def tertiary_trefoil_y(self):
        return self.Z[30]
        
    def tertiary_trefoil(self):
        return np.sqrt(self.Z[29]**2 + self.Z[30]**2)
        
    def quaternary_astig_x(self):
        return self.Z[31]
        
    def quaternary_astig_y(self):
        return self.Z[32]
        
    def quaternary_astig(self):
        return np.sqrt(self.Z[31]**2 + self.Z[32]**2)
        
    def quaternary_coma_x(self):
        return self.Z[33]
        
    def quaternary_coma_y(self):
        return self.Z[34]
        
    def quaternary_coma(self):
        return np.sqrt(self.Z[33]**2 + self.Z[34]**2)
        
    def quaternary_spherical(self):
        return self.Z[35]
        
    # =========================================================================
        
    def term(self, number=0, R=0, phi=0):
        '''one (unscaled) term of the zernike polynomial'''
        if number == 0:
            return 1
        elif number == 1:
            return R*np.cos(phi)
        elif number == 2:
            return R*np.sin(phi)
        elif number == 3:
            return (2*R**2 - 1)
        elif number == 4:
            return R**2*np.cos(2*phi)
        elif number == 5:
            return R**2*np.sin(2*phi)
        elif number == 6:
            return ((3*R**2-2)*R*np.cos(phi))
        elif number == 7:
            return ((3*R**2-2)*R*np.sin(phi))
        elif number == 8:
            return (6*R**4-6*R**2+1) 
        elif number == 9:
            return (R**3*np.cos(3*phi))
        elif number == 10:
            return (R**3*np.sin(3*phi))
        elif number == 11:
            return ((4*R**2-3)*R**2*np.cos(2*phi))
        elif number == 12:
            return ((4*R**2-3)*R**2*np.sin(2*phi))
        elif number == 13:
            return ((10*R**4-12*R**2+3)*R*np.cos(phi))
        elif number == 14:
            return ((10*R**4-12*R**2+3)*R*np.sin(phi))
        elif number == 15:
            return (20*R**6-30*R**4+12*R**2-1)
        elif number == 16:
            return (R**4*np.cos(4*phi))
        elif number == 17:
            return (R**4*np.sin(4*phi))
        elif number == 18:
            return ((5*R**2-4)*R**3*np.cos(3*phi))
        elif number == 19:
            return ((5*R**2-4)*R**3*np.sin(3*phi))
        elif number == 20:
            return ((15*R**4-20*R**2+6)*R**2*np.cos(2*phi))
        elif number == 21:
            return ((15*R**4-20*R**2+6)*R**2*np.sin(2*phi))
        elif number == 22:
            return ((35*R**6-60*R**4+30*R**2-4)*R*np.cos(phi))
        elif number == 23:
            return ((35*R**6-60*R**4+30*R**2-4)*R*np.sin(phi))
        elif number == 24:
            return (70*R**8-140*R**6+90*R**4-20*R**2+1)
        elif number == 25:
            return (R**5*np.cos(5*phi))
        elif number == 26:
            return (R**5*np.sin(5*phi))
        elif number == 27:
            return ((6*R**2-5)*R**4*np.cos(4*phi))
        elif number == 28:
            return ((6*R**2-5)*R**4*np.sin(4*phi))
        elif number == 29:
            return ((21*R**4-30*R**2+10)*R**3*np.cos(3*phi))
        elif number == 30:
            return ((21*R**4-30*R**2+10)*R**3*np.sin(3*phi))
        elif number == 31:
            return ((56*R**6-105*R**4+60*R**2-10)*R**2*np.cos(2*phi))
        elif number == 32:
            return ((56*R**6-105*R**4+60*R**2-10)*R**2*np.sin(2*phi))
        elif number == 33:
            return ((126*R**8-280*R**6+210*R**4-60*R**2+5)*R*np.cos(phi))
        elif number == 34:
            return ((126*R**8-280*R**6+210*R**4-60*R**2+5)*R*np.sin(phi))
        elif number == 35:
            return (252*R**10-630*R**8+560*R**6-210*R**4+30*R**2-1)
        else:
            raise ValueError('Invalid term specified. Must be integer within range 0 to 35')
        
    def coeffs(self, R, phi, num_terms=36):
        '''1D matrix of all (unscaled) coefficients for a given R and phi (used in least-squares optimization)'''
        mat = np.zeros(num_terms)
        for number in range(num_terms):
            mat[number] = self.term(number,R,phi)
        return mat
        
    def poly(self, R, phi, num_terms=36):
        '''Computed zernike polynomial using normalized coordinates R (radius) and p (angle)'''
        Z = np.zeros_like(R)
        for number in range(num_terms):
            Z += self.Z[number]*self.term(number,R,phi)
        return Z
        
    def rms(self, Z):
        return np.sqrt(np.nanmean(Z**2))
    
    #TODO - add units
    def info(self):
        for idt, term in enumerate(zernike.terms):
            print('%s: %.6f'%(term,self.Z[idt]))

    def show(self, projection='2d', num_pts=100):
        x = np.linspace(-1,1,num_pts)
        y = np.linspace(-1,1,num_pts)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        phi = np.arctan2(Y, X)
        Z = self.poly(R, phi, len(self.Z))
        Z[R>1] = np.nan
        
        if projection == '2d':
            fig, ax = plt.subplots()
            ax.imshow(Z, aspect='auto',interpolation='none')
        elif projection == '3d':
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.plot_surface(X,Y,Z)
        else:
            raise ValueError('Invalid projection type. Must be "2d" or "3d".')
        
        ax.set_title('Zernike Fringe Plot: RMS = %.3f'%self.rms(Z))
        ax.set_xlabel('Normalized X Coordinate')
        ax.set_ylabel('Normalized Y Coordinate')
        plt.axis('image')
        plt.show()
        
class fit(zernike):
    
    def __init__(self, wavefront, num_terms=36, remove_piston=True):

        x,y = zip(*wavefront.distribution.points)
        z = wavefront.phase_array()        
        
        self.x = np.array(x)
        self.y = np.array(y)
        self.z = np.array(z)
        self.num_terms = num_terms
        self.remove_piston = remove_piston
        
        self.R = np.sqrt(self.x**2 + self.y**2)
        self.phi = np.arctan2(self.y,self.x)
        self.num_pts = len(self.z)
        
        self.fit_coeffs()
        
    def _coeff_mat(self):
        '''coefficient matrix'''
        mat = np.zeros((self.num_pts,self.num_terms))
        for m in range(self.num_pts):
            mat[m,:] = self.coeffs(self.R[m],self.phi[m],self.num_terms)
        return mat
        
    def fit_coeffs(self):
        '''find best fit zernike terms for given X,Y,Z'''
        sol = np.linalg.lstsq(self._coeff_mat(),self.z)[0]
        self.Z = list(sol)
        if self.remove_piston:
            self.Z[0] = 0
            
    def residual(self):
        '''residual from zernike fit'''
        return self._coeff_mat().dot(self.Z) - self.z
        
    def show_residual(self):
        '''show the residual from the zernike fit'''
        res = self.residual()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.x,self.y,res)
        ax.set_title('Residual Plot: RMS = %.6f'%self.rms(res))
        ax.set_xlabel('Normalized X Coordinate')
        ax.set_ylabel('Normalized Y Coordinate')
        plt.axis('image')
        plt.show()

if __name__ == '__main__':
    
    from distribution import distribution
    from wavefront import wavefront
    import sample_lenses
        
    system = sample_lenses.Edmund_49_847()
    pupil_grid = distribution(type_='rectangular')

    system_wavefront = wavefront(system,pupil_grid,Hx=0,Hy=0)
    system_wavefront.show_phase()
    
    zernike_fit = fit(system_wavefront,num_terms=36,remove_piston=False)
    zernike_fit.show(projection='3d', num_pts=200)
    zernike_fit.show_residual()
    



