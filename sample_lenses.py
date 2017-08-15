# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:54:39 2017

@author: kramer
"""
import lens

def Edmund_49_847():
    '''define Edmund optics 49-847'''
    singlet = lens.lens(name='Edmund Optics: 49-847')
    
    # add surfaces
    singlet.add_surface(number=0, thickness=1e10, comment='object')
    singlet.add_surface(number=1, thickness=7, radius=19.93, stop=True, material='N-SF11')
    singlet.add_surface(number=2, thickness=21.48)
    singlet.add_surface(number=3, comment='image')
    
    # add aperture
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 25.4
    
    # add field
    singlet.field_type = 'angle'
    singlet.add_field(number=0, x=0, y=0)
    singlet.add_field(number=1, x=0, y=10)
    singlet.add_field(number=2, x=0, y=14)
    
    # add wavelength
    singlet.add_wavelength(number=0, value=486.1)
    singlet.add_wavelength(number=1, value=587.6, primary=True)
    singlet.add_wavelength(number=2, value=656.3)
    
    singlet.set_SA()
    
    return singlet
    
def double_Gauss():
    '''define double Gauss lens'''
    gauss = lens.lens(name='Double Gauss')
    
    # add surfaces
    gauss.add_surface(number=0, radius=1e10, thickness=1e10, comment='object')
    gauss.add_surface(number=1, radius=56.20238, thickness=8.75 , material='N-SSK2')
    gauss.add_surface(number=2, radius=152.28580, thickness=0.5)
    gauss.add_surface(number=3, radius=37.68262, thickness=12.5, material='N-SK2')
    gauss.add_surface(number=4, radius=1e10 , thickness=3.8 ,material='F5', manufacturer='schott')
    gauss.add_surface(number=5, radius=24.23130, thickness=16.369445)
    gauss.add_surface(number=6, radius=1e10, thickness=13.747957, stop=True)
    gauss.add_surface(number=7, radius=-28.37731, thickness=3.8, material='F5', manufacturer='schott')
    gauss.add_surface(number=8, radius=1e10, thickness=11, material='N-SK16')
    gauss.add_surface(number=9, radius=-37.92546, thickness=0.5)
    gauss.add_surface(number=10, radius=177.41176, thickness=7, material='N-SK16')
    gauss.add_surface(number=11, radius=-79.41143, thickness=61.487536)
    gauss.add_surface(number=12, comment='image')
    
    # add aperture
    gauss.aperture_type = 'imageFNO'
    gauss.aperture_value = 5
    
    # add field
    gauss.field_type = 'angle'
    gauss.add_field(number=0, x=0, y=0)
    gauss.add_field(number=1, x=0, y=10)
    gauss.add_field(number=2, x=0, y=14)
    
    # add wavelength
    gauss.add_wavelength(number=0, value=486.1)
    gauss.add_wavelength(number=1, value=587.6, primary=True)
    gauss.add_wavelength(number=2, value=656.3)
    
    gauss.set_SA()
    
    return gauss
    
def cooke_triplet():
    '''define a Cooke triplet'''
    cooke = lens.lens(name='Cooke Triplet: F/6.3')
    
    # add surfaces
    cooke.add_surface(number=0, thickness=1e10, comment='object')
    cooke.add_surface(number=1, thickness=3.7, radius=24.11, material='N-SK4')
    cooke.add_surface(number=2, thickness=4.66, radius=215.09)
    cooke.add_surface(number=3, thickness=1.6, radius=-94.81, material='J-F8')
    cooke.add_surface(number=4, thickness=2.37, radius=23.76)
    cooke.add_surface(number=5, thickness=6.76, stop=True)
    cooke.add_surface(number=6, thickness=3.5, radius=104.5, material='N-SK4')
    cooke.add_surface(number=4, thickness=84.126, radius=-63.89)
    cooke.add_surface(number=7, comment='image')
    
    # add aperture
    cooke.aperture_type = 'imageFNO'
    cooke.aperture_value = 6.3
    
    # add field
    cooke.field_type = 'angle'
    cooke.add_field(number=0, x=0, y=0)
    cooke.add_field(number=1, x=0, y=10)
    cooke.add_field(number=2, x=0, y=14)
    
    # add wavelength
    cooke.add_wavelength(number=0, value=486.1)
    cooke.add_wavelength(number=1, value=587.6, primary=True)
    cooke.add_wavelength(number=2, value=656.3)
    
    cooke.set_SA()
    
    return cooke
    
def microscope():
    '''define a microscope objective'''
    micro = lens.lens(name='Microscope Objective')
    
    micro.add_surface(number=0, thickness=1e10, comment='object')
    micro.add_surface(number=1, thickness=64.9, radius=553.260, material='N-FK51')
    micro.add_surface(number=1, thickness=4.4, radius=-247.644)
    micro.add_surface(number=1, thickness=59.4, radius=115.162, material='J-LLF2')
    micro.add_surface(number=1, thickness=17.6, radius=57.131)
    micro.add_surface(number=1, thickness=17.6, stop=True)
    micro.add_surface(number=1, thickness=74.8, radius=-57.646, material='SF5')
    micro.add_surface(number=1, thickness=77.0, radius=196.614, material='N-FK51')
    micro.add_surface(number=1, thickness=4.4, radius=-129.243)
    micro.add_surface(number=1, thickness=15.4, radius=2062.370, material='N-KZFS4')
    micro.add_surface(number=1, thickness=48.4, radius=203.781, material='CAF2')
    micro.add_surface(number=1, thickness=4.4, radius=-224.003)
    micro.add_surface(number=1, thickness=35.2, radius=219.864, material='CAF2')
    micro.add_surface(number=1, thickness=4.4, radius=793.3)
    micro.add_surface(number=1, thickness=26.4, radius=349.260, material='N-FK51')
    micro.add_surface(number=1, thickness=4.4, radius=-401.950)
    micro.add_surface(number=1, thickness=39.6, radius=91.992, material='N-SK11')
    micro.add_surface(number=1, thickness=96.189, radius=176.0)
    micro.add_surface(number=1, comment='image')
    
    # add aperture
    micro.aperture_type = 'imageFNO'
    micro.aperture_value = 0.9
    
    # add field
    micro.field_type = 'angle'
    micro.add_field(number=0, x=0, y=0)
    micro.add_field(number=1, x=0, y=0.7)
    micro.add_field(number=2, x=0, y=1)
    
    # add wavelength
    micro.add_wavelength(number=0, value=486.1)
    micro.add_wavelength(number=1, value=587.6, primary=True)
    micro.add_wavelength(number=2, value=656.3)
    
    micro.image_distance_solve()
    
    micro.set_SA()
    
    return micro
    
def aspheric_singlet():
    '''define an asphere'''
    asphere = lens.lens(name='Edmund Optics: 49-847')
    
    # add surfaces
    asphere.add_surface(number=0, thickness=1e10, comment='object')
    asphere.add_surface(number=1, thickness=6.61, semi_aperture=12.5, material='N-BK7', stop=True)
    k = -2.3087969
    asphere.add_surface(number=2, thickness=25.000356, semi_aperture=12.5, radius=-12.987, conic=k)
    asphere.add_surface(number=3, comment='image')
    
    # add aperture
    asphere.aperture_type = 'EPD'
    asphere.aperture_value = 25.4
    
    # add field
    asphere.field_type = 'angle'
    asphere.add_field(number=0, x=0, y=0)
    
    # add wavelength
    asphere.add_wavelength(number=0, value=532, primary=True)
    
    return asphere
    
def hubble():
    '''hubble space telescope'''
    asphere = lens.lens(name='Parabola')
    
    # add surfaces
    asphere.add_surface(number=0, thickness=0, comment='object')
    asphere.add_surface(number=1, thickness=-5, type_='aspheric', A=[-0.05], material='mirror', stop=True)
    asphere.add_surface(number=2, comment='image')
    
    # add aperture
    asphere.aperture_type = 'EPD'
    asphere.aperture_value = 25.4
    
    # add field
    asphere.field_type = 'angle'
    asphere.add_field(number=0, x=0, y=0)
    
    # add wavelength
    asphere.add_wavelength(number=0, value=532, primary=True)
    
    return asphere

if __name__ == '__main__':

    from visualize import plot2D
    cooke = double_Gauss()
    plot2D(cooke).show()











