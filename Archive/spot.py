# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 23:15:37 2017

@author: kramer

"""

from trace import real_raytrace

def spot(lens, pupil_distribution, H=0, P=0, wave=0):
    
    xs = []
    ys = []
    
    if lens.t[0] >= 1e10: # object at infinity
        for xv,yv in pupil_distribution.points:
            
            if lens.field_type == 'angle':
                pass
            x,y,z,K,L,Z = real_raytrace(lens,x0=xv,y0=yv,z0=0,K0=0,L0=0,M0=1,wave=wave,first_transfer=False)
            xs.append(x[-1])
            ys.append(y[-1])
    else:
        for xv,yv in pupil_distribution.points:
            
            
            xs.append(x[-1])
            ys.append(y[-1])
            
if __name__ == '__main__':
    
    from grid import distribution
    
    d = distribution(type_='hexapolar',)