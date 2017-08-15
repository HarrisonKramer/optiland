# -*- coding: utf-8 -*-
"""
raytrace.py
"""

import numpy as np

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
        

# ===========================================================================================================    
def transfer_meridional(lens):
    pass

def refract_meridional(lens):
    pass

def meridional_trace(lens):
    pass

# ===========================================================================================================
def transfer_skew(lens,surf,wave=0,x=0,y=1,z=0,K=0,L=0,M=1):
    '''transfer between two spherical surfaces'''
    t = lens.t[surf]
    C = lens.C[surf+1]
    e = t*M - (x*K + y*L + z*M)
    M1x = z + e*M - t
    M12 = x**2 + y**2 + z**2 - e**2 + t**2 - 2*t*z
    E1 = np.sqrt(M**2 - C*(C*M12 - 2*M1x))
    OPL = e + (C*M12 - 2*M1x)/(M + E1)
    x1 = x + OPL*K
    y1 = y + OPL*L
    z1 = z + OPL*M - t
    return x1, y1, z1, OPL*lens.n(wave)[surf]
    
def _transfer_from_ep(lens,wave=0,x=0,y=1,z=0,K=0,L=0,M=1):
    t = lens.EPL()
    C = lens.C[1]
    e = t*M - (x*K + y*L + z*M)
    M1x = z + e*M - t
    M12 = x**2 + y**2 + z**2 - e**2 + t**2 - 2*t*z
    E1 = np.sqrt(M**2 - C*(C*M12 - 2*M1x))
    OPL = e + (C*M12 - 2*M1x)/(M + E1)
    x1 = x + OPL*K
    y1 = y + OPL*L
    z1 = z + OPL*M - t
    return x1, y1, z1, OPL*lens.n(wave)[0]

def refract_skew(lens,surf,wave,x1=0,y1=1,z1=0,K=0,L=0,M=0):
    '''refract at spherical surface'''
    C = lens.C[surf]
    n0 = lens.n(wave)[surf-1]
    n1 = lens.n(wave)[surf]
    e = -(x1*K + y1*L + z1*M)
    M1x = z1 + e*M
    M12 = x1**2 + y1**2 + z1**2 - e**2
    E1 = np.sqrt(M**2 - C*(C*M12 - 2*M1x))
    Ep = np.sqrt(1 - (n0/n1)**2*(1 - E1**2))
    g = Ep - n0/n1*E1
    K1 = n0/n1*K - g*C*x1
    L1 = n0/n1*L - g*C*y1
    M1 = n0/n1*M - g*C*z1 + g
    return K1,L1,M1
    
# ===========================================================================================================
def transfer_aspheric(lens,surf,wave,x=0,y=1,z=0,K=0,L=0,M=1,tol=1e-6):
    '''transfer to aspheric surface'''
    C = lens.C[surf]
    x0,y0,z0,OPL = transfer_skew(lens,surf,wave,x,y,z,K,L,M)
    A = np.array(lens.surface_list[surf].A)
    
    # find intersection with asphere
    zb0 = 1e9
    while np.abs(zb0 - z0) > tol:
        s2 = x0**2 + y0**2
        zb0 = C*s2/(1 + np.sqrt(1 - C**2*s2)) + A*np.power(s2,range(1,len(A)+1))
        l0 = np.sqrt(1 - C**2*s2)
        m0 = -y0*(C + l0*(A*np.power(2,range(1,len(A)+1))))
        n0 = -x0*(C + l0*(A*np.power(2,range(1,len(A)+1))))
        G0 = l0*(zb0 - z0)/(K*l0 + L*m0 + M*n0)
        x0 += G0*K
        y0 += G0*L
        z0 += G0*M
    OPL += G0
        
    return x0,y0,z0,l0,m0,n0,OPL

# ===========================================================================================================
def refract_aspheric(lens,surf,wave,l0,m0,n0,x=0,y=1,z=0,K=0,L=0,M=0):
    '''refract at a real surface'''
    C = lens.C[surf]
    n0 = lens.n(wave)[surf-1]
    n1 = lens.n(wave)[surf]
    e = -(x*K + y*L + z*M)  
    M1z = z + e*M
    M12 = x**2 + y**2 + z**2 - e**2
    E1 = np.sqrt(M**2 - C*(C*M12 - 2*M1z))
    
    # refract
    Ep1 = np.sqrt(1 - (n0/n1)**2*(1 - E1**2))
    g1 = Ep1 - n0/n1*E1
    K1 = n0/n1*K - g1*C*x + g1
    L1 = n0/n1*L - g1*C*y
    M1 = n0/n1*M - g1*C*z
    
    return K1,L1,M1

# ===========================================================================================================
def real_raytrace(lens,Hx,Hy,Px,Py,wave,last='image',first=0,last_transfer=True):
    '''Perform a real (skew) ray trace of the optical system'''
    
    if last == 'image':
        last = len(lens.surface_list)-1
        
    surf_types = lens.surf_type
    
    xf = lens.x_field_list()
    yf = lens.y_field_list()
    
    xf_max = max(xf)
    yf_max = max(yf)
    
    EPD = lens.EPD()
    EPL = lens.EPL()
    
    OPL = 0
    
    # == find initial x,y,z,K,L,M =============================================
    if lens.t[0] >= 1e10:
        if lens.field_type is not 'angle':
            raise ValueError('Field type must be "angle" for an object at infinity.')
        
        x_ep = EPD*Px/2
        y_ep = EPD*Py/2
        z_ep = 0
        K0 =  np.cos(np.deg2rad(yf_max*Hy))*np.sin(np.deg2rad(xf_max*Hx))
        L0 = np.sin(np.deg2rad(yf_max*Hy))
        M0 = np.cos(np.deg2rad(yf_max*Hy))*np.cos(np.deg2rad(xf_max*Hx))
        if lens.surf_type[0] == 'standard':
            x0, y0, z0, OPL_new = _transfer_from_ep(lens,wave,x_ep,y_ep,z_ep,K0,L0,M0)
        elif lens.surf_type[0] == 'aspheric':
            pass
        
    else:
        if lens.field_type == 'angle':
            x_obj = (lens.t[0] - EPL)*np.tan(np.deg2rad(xf_max*Hx))
            y_obj = (lens.t[0] - EPL)*np.tan(np.deg2rad(yf_max*Hy))
        elif lens.field_type == 'object_height':
            x_obj = -xf_max*Hx
            y_obj = -yf_max*Hy
        else:
            raise ValueError('Invalid field type specified.')
        z_obj = 0
        dx = x_obj + Px*EPD/2
        dy = y_obj + Py*EPD/2
        dz = lens.t[0] - EPL
        mag = np.sqrt(dx**2 + dy**2 + dz**2)
        K0 = dx/mag
        L0 = dy/mag
        M0 = dz/mag
        
        if lens.surf_type[1] == 'standard':
            x0, y0, z0, OPL_new = transfer_skew(lens,1,wave,x_obj,y_obj,z_obj,K0,L0,M0)
        elif lens.surf_type[1] == 'aspheric':
            x0,y0,z0,l0,m0,n0,OPL_new = transfer_aspheric(lens,1,wave,x_obj,y_obj,z_obj,K0,L0,M0)
        
    OPL += OPL_new
    
    x = [x0]
    y = [y0]
    z = [z0]
    K = [K0]
    L = [L0]
    M = [M0]
    
    # =========================================================================
        
    for surf in range(first+1,last):
        
        # refraction
        if surf_types[surf] == 'standard':
            Knew,Lnew,Mnew = refract_skew(lens,surf,wave,x[-1],y[-1],z[-1],K[-1],L[-1],M[-1])
        if surf_types[surf] == 'aspheric' or lens.surface_list[surf].conic != 0:
            Knew,Lnew,Mnew = refract_aspheric(lens,surf,wave,l0,m0,n0,x[-1],y[-1],z[-1],K[-1],L[-1],M[-1])
    
        K.append(Knew)
        L.append(Lnew)
        M.append(Mnew)
        
        # transfer
        if surf!= last-1 or last_transfer:
            if surf_types[first] == 'standard':
                xnew,ynew,znew,OPL_new = transfer_skew(lens,surf,wave,x[-1],y[-1],z[-1],K[-1],L[-1],M[-1])
            elif surf_types[first] == 'aspheric' or lens.surface_list[surf].conic != 0:
                x0,y0,z0,l0,m0,n0,OPL_new = transfer_aspheric(lens,surf,wave,x_obj,y_obj,z_obj,K0,L0,M0)

            x.append(xnew)
            y.append(ynew)
            z.append(znew)
            
            OPL += OPL_new
        
    return x,y,z,K,L,M,OPL
    
if __name__ == '__main__':
    
    from lens import lens
    
    singlet = lens(name='Singlet')
    
    # add surfaces
    singlet.add_surface(number=0, thickness=200, comment='object')
    singlet.add_surface(number=1, thickness=15, radius=50, stop=True, material='N-SF11')
    singlet.add_surface(number=2, thickness=147.368421, radius=-50)
    singlet.add_surface(number=3, comment='image')
    
    # add aperture
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    # add field
    singlet.field_type = 'angle'
    singlet.add_field(number=0, x=0, y=0)
    singlet.add_field(number=1, x=0, y=10)
    singlet.add_field(number=2, x=0, y=14)
    
    # add wavelength
    singlet.add_wavelength(number=0, value=486.1)
    singlet.add_wavelength(number=1, value=587.6, primary=True)
    singlet.add_wavelength(number=2, value=656.3)

    x,y,z,K,L,M,OPL = real_raytrace(lens=singlet,wave=0,Hx=0,Hy=1,Px=0,Py=1)
    print(x)




































