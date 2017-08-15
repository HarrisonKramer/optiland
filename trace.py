# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 21:47:46 2017

@author: kramer
"""

import numpy as np
from ray import ray
from transform import transform
from transfer import transfer
from refract import refract

#TODO - update to class based like real ray tracing
# ===========================================================================================================
def paraxial_trace(lens,y0,u0,wave,last='image',first=0,reverse=False,first_transfer=True,last_transfer=True):
    """
    paraxial_raytrace: Performs a paraxial ray trace through an optical system in forward or
                reverse direction and between specified surfaces. Furthermore, 
                whether first/last operation should be a ray transfer can be specified.
                
    Inputs:
        y0 - input ray height at start surface
        u0 - input ray angle leaving start surface
        reverse - boolean, whether raytrace is reversed (default=False)
        first - first surface of trace (default=0)
        last - last surface of trace (default='image', which signifies image surface)
        first_transfer - whether to transfer from first surface (note: 
                        if False, for forward traces, first operation is 
                        refraction at surface first+1). (default=True)
        last_transfer - whether to transfer to last surface (note: if False,
                        for reverse traces, first operation is refraction
                        at surface last-1). (default=True)
                        
    Outputs:
        y - ray heights in order from first to last (i.e. order same for forward and reverse traces)
        u - ray angles in order from first to last (i.e. order same for forward and reverse traces)
        
    Usage:
    
        y, u = raytrace(1,0)
        y, u = raytrace(5,0.15,firstTransfer=False,lastTransfer=False)
        y, u = raytrace(0,0.05,reverse=True,first=2,last=7,firstTransfer=False,lastTransfer=True)
    """
    
    if last == 'image':
        last = len(lens.surface_list) - 1
    
    y = [y0]
    u = [u0]
    
    if reverse:
        for k in range(last-1,first,-1):
            
            n0 = lens.n(wave)[k-1]
            n1 = lens.n(wave)[k]
            t = lens.t[k]
            R = lens.R[k]
            P = (n1 - n0) / R
            
            if k != last-1 or last_transfer:
                y.append( y[-1] - u[-1]*t )
            u.append( 1/n0*(n1*u[-1] + y[-1]*P) )
        if first_transfer:
            y.append( y[-1] - u[-1]*lens.surface_list[first].thickness )
        y.reverse()
        u.reverse()
        
    else:
        
        if first_transfer:
            y.append(y[0] + lens.surface_list[first].thickness*u[0])
        for k in range(first,last-1):
            
            n0 = lens.n(wave)[k]
            n1 = lens.n(wave)[k+1]
            t = lens.surface_list[k+1].thickness
            R = lens.surface_list[k+1].radius
            P = (n1 - n0) / R
            
            u.append( (1/n1)*((n0*u[-1]) - y[-1]*P))
            if k!= last-1 or last_transfer:
                y.append( y[-1] + u[-1]*t )
        
    return y, u

# =========================================================================================================== 
def paraxial_trace_HP(lens,Hx,Hy,Px,Py,wave):
    H = np.sqrt((Hx/2)**2 + (Hy/2)**2)
    P = np.sqrt((Px/2)**2 + (Py/2)**2)
    xf = np.array(lens.x_field_list())
    yf = np.array(lens.y_field_list())
    f = np.sqrt(xf**2 + yf**2)
    EPD = lens.EPD()
    EPL = lens.EPL()
    if lens.t[0] >= 1e10:    # field type is angle if object at infinity
        if lens.field_type != 'angle':
            raise ValueError('Field type must be angle for object at infinity.')
        u0 = np.tan(H*max(np.deg2rad(f)))
    else:
        if lens.field_type == 'angle':
            dy = (lens.t[0]-EPL)*H*max(f) + P*EPD/2
        elif lens.field_type == 'object_height':
            dy = H*max(f) + P*EPD/2
        dx = lens.t[0] - EPL
        u0 = dy/dx
    y0 = P*EPD/2 + EPL*u0
        
    y, u = paraxial_trace(lens, y0, u0, wave, first_transfer=False)
    return y, u        

# ========================================================================================
class trace(object):
    
    def __init__(self, lens, Hx, Hy, Px, Py, wave):
        self.lens = lens
        self.Hx = Hx
        self.Hy = Hy
        self.Px = Px
        self.Py = Py
        self.wave = wave
        self.ray_list = []
        self._HP_to_ray()
        
    def _HP_to_ray(self):
        '''Convert normalized field and pupil coord's to ray x,y,z coordinates and direction cosines K,L,M
           and trace from object plane (or entrance pupil if object at infinity) to first surface
        '''
        xf = self.lens.x_field_list()
        yf = self.lens.y_field_list()
        xf_max = max(xf)
        yf_max = max(yf)
        EPD = self.lens.EPD()
        EPL = self.lens.EPL()
        
        if self.lens.t[0] >= 1e10:
            if self.lens.field_type is not 'angle':
                raise ValueError('Field type must be "angle" for an object at infinity.')
            
            x0 = EPD*self.Px/2
            y0 = EPD*self.Py/2
            z0 = 0
            K0 =  np.cos(np.deg2rad(yf_max*self.Hy))*np.sin(np.deg2rad(xf_max*self.Hx))
            L0 = np.sin(np.deg2rad(yf_max*self.Hy))
            M0 = np.cos(np.deg2rad(yf_max*self.Hy))*np.cos(np.deg2rad(xf_max*self.Hx))
            ray_in = ray(x0,y0,z0,K0,L0,M0)
            self.ray_list.append(ray_in)
            
            if self.lens.surface_list[1].decentered or self.lens.surface_list[1].tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], self.lens.surface_list[1].tilt,
                                              self.lens.surface_list[1].decenter).run()  
            
            ray_new = transfer(self.lens.surface_list[0], self.lens.surface_list[1], ray_in, 
                               self.wave).run(EP_transfer=True, t=EPL)
                               
            if self.lens.surface_list[1].decentered or self.lens.surface_list[1].tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], self.lens.surface_list[1].tilt,
                                              self.lens.surface_list[1].decenter).unrun()  
            
        else:
            if self.lens.field_type == 'angle':
                x0 = (self.lens.t[0] - EPL)*np.tan(np.deg2rad(xf_max*self.Hx))
                y0 = (self.lens.t[0] - EPL)*np.tan(np.deg2rad(yf_max*self.Hy))
            elif self.lens.field_type == 'object_height':
                x0 = xf_max*self.Hx
                y0 = yf_max*self.Hy
            else:
                raise ValueError('Invalid field type specified.')
            z0 = 0
            dx = x0 + self.Px*EPD/2
            dy = y0 + self.Py*EPD/2
            dz = self.lens.t[0] - EPL
            mag = np.sqrt(dx**2 + dy**2 + dz**2)
            K0 = dx/mag
            L0 = dy/mag
            M0 = dz/mag
            ray_in = ray(-x0,-y0,z0,K0,L0,M0)
            self.ray_list.append(ray_in)
            
            if self.lens.surface_list[1].decentered or self.lens.surface_list[1].tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], self.lens.surface_list[1].tilt,
                                              self.lens.surface_list[1].decenter).run()  
                
            ray_new = transfer(self.lens.surface_list[0], self.lens.surface_list[1], ray_in, 
                               self.wave).run()
                               
            if self.lens.surface_list[1].decentered or self.lens.surface_list[1].tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], self.lens.surface_list[1].tilt,
                                              self.lens.surface_list[1].decenter).unrun()  
            
        self.ray_list.append(ray_new)
        return self
        
    def run(self):
        
        max_surf = len(self.lens.surface_list) - 1
        for ids in range(1,max_surf):
            
            surf0 = self.lens.surface_list[ids-1]
            surf1 = self.lens.surface_list[ids]
            surf2 = self.lens.surface_list[ids+1]
            
            # transform if tilted or decentered surface
            if surf1.decentered or surf1.tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], surf1.tilt, surf1.decenter).run()
            
            # == refract ==================================================
            ray_new = refract(surf0, surf1, self.ray_list[-1], self.wave).run()
            self.ray_list.append(ray_new)
            
            # remove transform
            if surf1.decentered or surf1.tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], surf1.tilt, surf1.decenter).unrun()
            
            if surf2.decentered or surf2.tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], surf2.tilt, surf2.decenter).run()            
            
            # == transfer ==================================================
            ray_new = transfer(surf1, surf2, self.ray_list[-1], self.wave).run()
            self.ray_list.append(ray_new)
            
            if surf2.decentered or surf2.tilted:
                self.ray_list[-1] = transform(self.ray_list[-1], surf2.tilt, surf2.decenter).unrun()        
                
        return self
        
    @property
    def x(self):
        x = [self.ray_list[0].x]
        for k in range(1,len(self.ray_list),2):
            x.append(self.ray_list[k].x)
        return np.array(x) + self.lens.decenter[0]
    
    @property
    def y(self):
        y = [self.ray_list[0].y]
        for k in range(1,len(self.ray_list),2):
            y.append(self.ray_list[k].y)
        return np.array(y) + self.lens.decenter[1]
    
    @property
    def z(self):
        z = [self.ray_list[0].z]
        for k in range(1,len(self.ray_list),2):
            z.append(self.ray_list[k].z)
        return np.array(z)
    
    @property
    def K(self):
        K = []
        for k in range(0,len(self.ray_list),2):
            K.append(self.ray_list[k].K)
        return K
    
    @property
    def L(self):
        L = []
        for k in range(0,len(self.ray_list),2):
            L.append(self.ray_list[k].L)
        return L
    
    @property
    def M(self):
        M = []
        for k in range(0,len(self.ray_list),2):
            M.append(self.ray_list[k].M)
        return M
    
    @property
    def OPL(self):
        OPL = []
        for k in range(1,len(self.ray_list),2):
            OPL.append(self.ray_list[k].OPL)
        return OPL
        
    def info(self):
        
        for ids, rayi in enumerate(self.ray_list):
            print('--- Ray %d ---'%ids)
            rayi.info()
        return self


if __name__ == '__main__':
    
    from visualize import plot2D
    import lens
    
    singlet = lens.lens(name='Singlet')
    
    # add surfaces
    singlet.add_surface(number=0, thickness=1e10, comment='object')
    singlet.add_surface(number=1, thickness=7, radius=19.93, decenter=[0,0], stop=True, material='N-SF11')
    singlet.add_surface(number=2, thickness=21.48, decenter=[0,0])
    singlet.add_surface(number=3, comment='image')
    
    # add aperture
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    # add field
    singlet.field_type = 'angle'
    singlet.add_field(number=0, x=0, y=0)
    singlet.add_field(number=0, x=0, y=5)
    
    # add wavelength
    singlet.add_wavelength(number=0, value=486.1)
    singlet.add_wavelength(number=1, value=587.6, primary=True)
    singlet.add_wavelength(number=2, value=656.3)
    
    singlet.image_solve = True
    singlet.update_paraxial()
    
    plot2D(singlet,num_pupil_rays=10).show()
    
    raytrace = trace(singlet, Hx=0, Hy=0, Px=0, Py=1, wave=0).run()
#    raytrace.info()



















