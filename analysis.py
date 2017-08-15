# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from trace import trace, paraxial_trace_HP
import numpy as np
from distribution import distribution
from wavefront import wavefront

# ===========================================================================================
def ray_fan_X_single(lens, Hx=0, Hy=0, wave=0, **options):
    
    num_pts = 25
    xs = []
    px = np.linspace(0,1,num_pts)
    yp,up = paraxial_trace_HP(lens,Hx=Hx,Hy=Hy,Px=0,Py=0,wave=wave)
    for pt in px:
        raytrace = trace(lens,Hx=Hx,Hy=Hy,Px=pt,Py=0,wave=wave)
        raytrace.run()
        xs.append(raytrace.x[-1] - yp[-1])
    
    xs = [x-xs[0] for x in xs]  # remove distortion
    
    return px, xs

# ===========================================================================================
def ray_fan_Y_single(lens, Hx=0, Hy=0, wave=0, **options):

    num_pts = 50
    ys = []
    py = np.linspace(-1,1,num_pts)
    yp,up = paraxial_trace_HP(lens,Hx=Hx,Hy=Hy,Px=0,Py=0,wave=wave)
    for pt in py:
        raytrace = trace(lens,Hx=Hx,Hy=Hy,Px=0,Py=pt,wave=wave)
        raytrace.run()
        ys.append(raytrace.y[-1] - yp[-1])
        
    raytrace = trace(lens,Hx=Hx,Hy=Hy,Px=0,Py=0,wave=wave).run()
    ys = [y-raytrace.y[-1]+yp[-1] for y in ys] # remove distortion
    
    return py, ys

# ===========================================================================================
def ray_fan_X(lens, Hx=0, Hy=0):
    
    fig, ax = plt.subplots()
    
    colors = ['r','g','b','c','m','y']
    for idw, wave in enumerate(lens.wavelengths_list):
        px, xs = ray_fan_X_single(lens, Hx=Hx, Hy=Hy, wave=idw)
        ax.plot(px, xs, colors[idw%6], label='%.1f nm'%wave)
        
    ax.legend(loc='best')
    plt.xlabel('Pupil Y (%s)'%lens.length_unit)
    plt.ylabel('Transverse Ray Error X (%s)'%lens.length_unit)
    plt.title('%s: Ray Aberration Fan X, Hx: %.1f, Hy: %.1f'%(lens.name, Hx, Hy))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    
# ===========================================================================================
def ray_fan_Y(lens, Hx=0, Hy=0):
    
    fig, ax = plt.subplots()
    
    colors = ['r','g','b','c','m','y']
    for idw, wave in enumerate(lens.wavelengths_list):
        py, ys = ray_fan_Y_single(lens, Hx=Hx, Hy=Hy, wave=idw)
        ax.plot(py, ys, colors[idw%6], label='%.1f nm'%wave)
        
    ax.legend(loc='best')
    plt.xlabel('Pupil Y (%s)'%lens.length_unit)
    plt.ylabel('Transverse Ray Error Y (%s)'%lens.length_unit)
    plt.title('%s: Ray Aberration Fan Y, Hx: %.1f, Hy: %.1f'%(lens.name, Hx, Hy))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    
# ===========================================================================================
def ray_fan(lens, Hx=0, Hy=0):
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    
    colors = ['r','g','b','c','m','y']
    for idw, wave in enumerate(lens.wavelengths_list):
        px, xs = ray_fan_X_single(lens, Hx=Hx, Hy=Hy, wave=idw)
        py, ys = ray_fan_Y_single(lens, Hx=Hx, Hy=Hy, wave=idw)
        ax1.plot(py, ys, colors[idw%6], label='%.1f nm'%wave)
        ax2.plot(px, xs, colors[idw%6], label='%.1f nm'%wave)
        
    ax1.set_title('Ray Fan Y, Hx: %.1f, Hy: %.1f'%(Hx, Hy))
    ax2.set_title('Ray Fan X, Hx: %.1f, Hy: %.1f'%(Hx, Hy))
    ax1.axhline(y=0, color='k')
    ax1.axvline(x=0, color='k')
    ax2.axhline(y=0, color='k')
    ax2.axvline(x=0, color='k')
    ax1.set_xlabel('Pupil Coordinate Y (%s)'%lens.length_unit)
    ax1.set_ylabel('Transverse Ray Error Y (%s)'%lens.length_unit)
    ax2.set_xlabel('Pupil Coordinate X (%s)'%lens.length_unit)
    ax2.set_ylabel('Transverse Ray Error X (%s)'%lens.length_unit)

# ===========================================================================================
def spot_diagram_standard_single(lens, pupil_distribution, Hx, Hy, wave=0, **options):
    
    xs = []
    ys = []
    for Px,Py in pupil_distribution.points:
        raytrace = trace(lens,Hx,Hy,Px,Py,wave).run()
        xs.append(raytrace.x[-1])
        ys.append(raytrace.y[-1])
        
    return xs, ys
    
def spot_rms(xs,ys):
    x_avg = np.mean(xs)
    y_avg = np.mean(ys)
    rms = np.sqrt(np.mean((xs-x_avg)**2 + (ys-y_avg)**2))
    return rms

def spot_geometric_radius(xs,ys):
    x_avg = np.mean(xs)
    y_avg = np.mean(ys)
    r_geo = np.sqrt( np.max(np.abs(xs-x_avg))**2 + np.max(np.abs(ys-y_avg))**2 )
    return r_geo
    
def spot_diagram_standard(lens, pupil_distribution, Hx, Hy, airy=False):
    
    fig, ax = plt.subplots()

    colors = ['r+','gx','b*','cd','mv','yh']
    geo = []
    for idw, wave in enumerate(lens.wavelengths_list):
        xs = []
        ys = []
        for Px,Py in pupil_distribution.points:
            raytrace = trace(lens,Hx,Hy,Px,Py,wave=idw).run()
            xs.append(raytrace.x[-1])
            ys.append(raytrace.y[-1])
        ax.plot(xs,ys,colors[idw%6],label='%.1f nm'%wave)
        geo.append(spot_geometric_radius(xs,ys))
        rad_str = 'Wave: %.1f nm, RMS Radius: %.3f mm, Geo. Radius: %.3f mm'%(wave,spot_rms(xs,ys),geo[-1])
        print(rad_str)
                
    if airy:
        theta = np.linspace(0,2*np.pi,100)
        r = 1.22*lens.wavelengths_list[lens.prim_wave]/1e6*lens.FNO()   
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        ax.plot(x,y,'k')
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    plt.xlabel('Image Plane X (%s)'%lens.length_unit)
    plt.ylabel('Image Plane Y (%s)'%lens.length_unit)
    plt.title('%s, Spot Diagram: Hx=%.1f, Hy=%.1f'%(lens.name.title(),Hx, Hy))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal', adjustable='box')
    plt.show()

# TODO - complete
# ===========================================================================================
def spot_diagram_through_focus(lens, pupil_distribution, Hx, Hy, total_span=3):
    
    fig, axs = plt.subplots(1,3)
    
    axs = axs.ravel()
    
    spans = np.linspace(-total_span/2,total_span/2,3)
    for span, k in enumerate(spans):
        
        lens.surface_list[-2].thickness += span
        
        # perform spot diagram
        xs, ys = spot_diagram_standard_single(lens, pupil_distribution, Hx, Hy)
        axs[k].plot(xs,ys,'*')
        
        lens.surface_list[-2].thickness -= span
        
    plt.show()

# ===========================================================================================
def distortion_single(lens, type_='f_tan_theta', wave=0, **options):
    if lens.t[0] < 1e10:
        raise ValueError('Distortion plot requires lens having infinite object.')
    
    num_pts = 25
    Hys = np.linspace(0,1,num_pts)
    yf = np.array(lens.y_field_list())
    max_f = max(yf)
    
    if type_ == 'f_tan_theta':
        yp = lens.f2()*np.tan(Hys*np.deg2rad(max_f))
    elif type_ == 'f_theta':
        yp = lens.f2()*Hys*np.deg2rad(max_f)
    else:
        raise ValueError('Invalid distortion type selected.')
        
    yr = []
        
    for Hy in Hys:
        raytrace = trace(lens,Hx=0,Hy=Hy,Px=0,Py=0,wave=wave).run()
        yr.append(raytrace.y[-1])

    dist = [100*(x-y)/y for x,y in zip(yr,yp)]
    return dist, Hys

# ===========================================================================================  
def distortion(lens, type_='f_tan_theta', **options):
    fig, ax = plt.subplots()
    
    colors = ['r','g','b','c','m','y']
    for idw, wave in enumerate(lens.wavelengths_list):
        dist, Hys = distortion_single(lens, type_=type_, wave=idw)
        ax.plot(dist, Hys, colors[idw%6], label='%.1f nm'%wave)
        
    plt.xlabel('Distortion (%)')
    plt.ylabel('Normalize Y Field')
    plt.title('%s: Distortion'%lens.name)
    plt.legend(loc='best')
    plt.show()

# ===========================================================================================
def grid_distortion(lens, num_pts=100, type_='f_tan_theta', wave=0, **options):
    pass

# ===========================================================================================
def field_curvature(lens):
    pass

# ===========================================================================================
def y_ybar(lens,wave=0):
    
    fig, ax = plt.subplots()
    ya, ua, = paraxial_trace_HP(lens,Hx=0,Hy=0,Px=0,Py=1,wave=wave)
    yb, ub, = paraxial_trace_HP(lens,Hx=0,Hy=1,Px=0,Py=0,wave=wave)
    
    ax.plot(yb,ya)
    ax.plot(yb,ya,'*')
    plt.ylabel('Marginal Ray (%s)'%lens.length_unit)
    plt.xlabel('Chief Ray (%s)'%lens.length_unit)
    plt.title('%s: Y/Ybar Diagram'%lens.name)
    plt.show()
    
# ===========================================================================================
def OPD_fan(lens, Hx=0, Hy=0, wave=0):
    '''OPD fan in X and Y for a given field and wavelength'''
    gridX = distribution(type_='lineX',num_pts=50)
    gridY = distribution(type_='lineY',num_pts=50)
    X_wavefront = wavefront(lens,gridX,Hx=Hx,Hy=Hy,wave=0)
    Y_wavefront = wavefront(lens,gridY,Hx=Hx,Hy=Hy,wave=0)
    f, axarr = plt.subplots(1,2)
    axarr[0].plot(gridY.y, Y_wavefront.phase_array())
    axarr[0].set_title('OPD Fan Y: Hx=%.1f, Hy=%.1f'%(Hx, Hy))
    axarr[0].set_xlabel('Normalized Pupil Coordinate Y')
    axarr[0].set_ylabel('OPD Y (radians)')
    axarr[0].axhline(y=0, color='k')
    axarr[0].axvline(x=0, color='k')
    axarr[1].plot(gridY.y, X_wavefront.phase_array())
    axarr[1].set_title('OPD Fan X: Hx=%.1f, Hy=%.1f'%(Hx, Hy))
    axarr[1].set_xlabel('Normalized Pupil Coordinate X')
    axarr[1].set_ylabel('OPD X (radians)')
    axarr[1].axhline(y=0, color='k')
    axarr[1].axvline(x=0, color='k')
    plt.show()
    
# ===========================================================================================
def mtf_geometric_single(lens, v0, Hx=0, Hy=0, wave=0, N=300):
    
    # define pupil distribution for raytrace
    dist = distribution(type_='rectangular', num_pts=500)
    
    # raytrace
    xs = []
    ys = []
    for Px,Py in dist.points:
        raytrace = trace(lens,Hx,Hy,Px,Py,wave).run()
        xs.append(raytrace.x[-1])
        ys.append(raytrace.y[-1])
        
    # find line spread function by integrating along x using histogram
    rad_geo = spot_geometric_radius(xs, ys)
    dx = 2*rad_geo/N
    bins = np.arange(-rad_geo,rad_geo,dx)
    A, x = np.histogram(xs, bins)
    v = np.linspace(0,v0,len(A))
    
    # x is bin edges, use the upper edge of bin
    x = x[1:]
    
    # calculate MTF at different cycles/mm
    denom = np.sum(A*dx)
    MTF = np.zeros_like(v)
    for k in range(len(v)):
        
        numC = np.sum(A * np.cos(2*np.pi*v[k]*x) * dx)
        numS = np.sum(A * np.sin(2*np.pi*v[k]*x) * dx)
        Ac = numC/denom
        As = numS/denom
        MTF[k] = np.sqrt(Ac**2 + As**2)
        
    return v,MTF
    
def mtf_geometric(lens, wave=0, N=300):
    '''Reference: Smith, Modern Optical Engineering, Section 11.9,
       Computation of the Modulation Transfer Function'''
       
    fig, ax = plt.subplots()
       
    # find cutoff frequency (assumes circular aperture)
    NA = lens.imageNA()
    L = lens.wavelengths_list[wave]/1e6
    v0 = 2*NA/L     # cutoff frequency
    
    colors = ['r','g','b','c','m','y']
    for idx, field in enumerate(lens.fields_list):
        Hx,Hy = field
        
        if max(lens.x_field_list()) == 0:
            Hx = 0
        else:
            Hx /= max(lens.x_field_list())
            
        if max(lens.y_field_list()) == 0:
            Hy = 0
        else:
            Hy /= max(lens.y_field_list())
            
        v,MTF = mtf_geometric_single(lens, v0, Hx=Hx, Hy=Hy, wave=wave, N=N)
        ax.plot(v,MTF,colors[idx%6],label='Hx=%.2f, Hy=%.2f'%(Hx,Hy))
    
    # find diffraction limited MTF
    phi = np.arccos(L*v/2/NA)
    MTF_dl = 2/np.pi*(phi - np.cos(phi)*np.sin(phi))
        
    # plot result
    ax.plot(v,MTF_dl,'k',label='Diffraction Limit')
    ax.legend(loc='best')
    ax.set_xlim([0,v0])
    plt.ylabel('Modulation Transfer Function')
    plt.xlabel('v (lp/mm)')
    plt.title('Geometric MTF')
    plt.grid()  
    plt.show()

if __name__ == '__main__':
    
    import sample_lenses
    
    system = sample_lenses.cooke_triplet()
    
    mtf_geometric(system, wave=0)
    
    
    
    
    
    