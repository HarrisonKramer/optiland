# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 20:46:27 2017

@author: kramer
"""
import numpy as np
import matplotlib.pyplot as plt
from trace import trace

class plot2D(object):
    
    def __init__(self, lens, **options):
        self.__dict__.update(options)
        self.lens = lens
        
        if not hasattr(self,'num_pupil_rays'):
            self.num_pupil_rays = 5
            
        if not hasattr(self,'num_sag_pts'):
            self.num_sag_pts = 25
            
    def init_plot(self):
        '''initialize plot and define variables'''
        self.fig, self.ax = plt.subplots()
        self.ax.set_autoscale_on(False)
        
        self.N = self.lens.num_surfaces
        self.SA = self.lens.SA
        self.C = self.lens.C
        self.n = self.lens.n(self.lens.prim_wave)
        self.F = self.lens.y_field_list()
        
        self.y_list = self.lens.y_field_list()
        if max(self.y_list) is 0:
            self.Hy_all = [0 for y in self.y_list]
        else:
            self.Hy_all = [y/max(self.y_list) for y in self.y_list]
        
        self.sag_max = np.zeros(self.N-1)
        self.sag_min = np.zeros(self.N-1)
        
    def surface_vertices(self):
        '''find vertices of all surfaces'''
        self.t = np.zeros(self.N)
        self.t[1:] = self.lens.t[:-1]
        if self.t[1]>= 1e10:
            self.t[1] = 10
        self.t = np.cumsum(self.t)
            
    def set_axes(self):
        '''set axes limits'''
        self.ax.set_xlim([min(self.t), max(self.t)])
        self.ax.set_ylim([-1.5*max(self.SA[1:]), 1.5*max(self.SA[1:])])
    
    def sag(self, C, SA, k, **kwargs):
        '''find sag of a surface'''
        if 'A' in kwargs:
            A = kwargs['A']
        else:
            A = [0]
        y = np.linspace(-SA,SA,self.num_sag_pts)
        z = C*y**2/(1 + np.sqrt(1 - y**2*C**2*(k+1))) + \
            sum([(y**2)**(m+1)*a for m,a in enumerate(A)])
        return y, z
        
    def decenter_surface(self, y, decenter):
        return y + decenter
    
    # TODO - complete
    def tilt_surface(self):
        pass
        
    def draw_surface(self, surf):
        '''draw single surface in yz plane'''
        if self.lens.surf_type[surf] == 'standard':
            y, z = self.sag(self.C[surf], self.SA[surf], k=0)
        elif self.lens.surf_type[surf] == 'aspheric':
            y, z = self.sag(C=self.C[surf], SA=self.SA[surf], 
                            k=self.lens.surface_list[surf].conic, A=self.lens.surface_list[surf].A)
                            
        y = self.decenter_surface(y,self.lens.surface_list[surf].decenter[1]) 
                            
        self.sag_max[surf-1] = z[-1]
        self.sag_min[surf-1] = z[0]
        self.ax.plot(z+self.t[surf],y,'k')
    
    def draw_all_surfaces(self):
        '''draw all surfaces, excluding object'''
        for surf in range(1,self.N):
            self.draw_surface(surf)
    
    def decenter_SA(self, surf, side):
        '''add y decenter to upper or lower SA for a given surface'''
        if side == 'upper':
            return self.SA[surf] + self.lens.surface_list[surf].decenter[1]
        elif side =='lower':
            return -self.SA[surf] + self.lens.surface_list[surf].decenter[1]
        else:
            return None
        
    # TODO - complete
    def tilt_SA(self):
        pass
    
    def draw_edge(self, surf):
        '''draw the edges of the one surface'''
        pass

        if abs(self.n[surf]) != 1:
            x_upper = [self.t[surf]+self.sag_max[surf-1],self.t[surf+1]+self.sag_max[surf]]
            x_lower = [self.t[surf]+self.sag_min[surf-1],self.t[surf+1]+self.sag_min[surf]]
            y_upper = [self.decenter_SA(surf,'upper'),self.decenter_SA(surf+1,'upper')]
            y_lower = [self.decenter_SA(surf,'lower'),self.decenter_SA(surf+1,'lower')]
            
            if self.SA[surf+1] > self.SA[surf]:
                self.ax.plot([x_upper[0],x_upper[0]],y_upper,'k')
                self.ax.plot([x_lower[0],x_lower[0]],y_lower,'k')
                self.ax.plot(x_upper,[y_upper[1],y_upper[1]],'k')
                self.ax.plot(x_lower,[y_lower[1],y_lower[1]],'k')
                
            elif self.SA[surf+1] < self.SA[surf]:
                self.ax.plot([x_upper[1],x_upper[1]],y_upper,'k')
                self.ax.plot([x_lower[1],x_lower[1]],y_lower,'k')
                self.ax.plot(x_upper,[y_upper[0],y_upper[0]],'k')
                self.ax.plot(x_lower,[y_lower[0],y_lower[0]],'k')
                
            else:                           
                self.ax.plot(x_upper,y_upper,'k')
                self.ax.plot(x_lower,y_lower,'k')
    
    def draw_all_edges(self):
        '''draw all edges of all glass'''
        for surf in range(1,self.N):
            self.draw_edge(surf)
    
    def trace_rays(self, Hy, Py):
        raytrace = trace(self.lens,0,Hy,0,Py,wave=self.lens.prim_wave).run()
        return raytrace.y, raytrace.z
    
    def plot_rays(self, Hy, color='k'):
        Py_all = np.linspace(-1,1,self.num_pupil_rays)
        label_flag = True
        for Py in Py_all:
            y,z = self.trace_rays(Hy,Py)
            z_plot = z + self.t
            y_plot = y
            if self.lens.field_type == 'angle':
                y_plot[0] = y_plot[1] - z_plot[1]*np.tan(np.deg2rad(Hy*max(self.y_list)))
            elif self.lens.field_type == 'object_height':
                y_plot[0] = -Hy*max(self.y_list)
            if label_flag:
                self.ax.plot(z_plot,y_plot,color,label='Field Y: %.1f '%(Hy*max(self.y_list)))
                label_flag = False
            else:
                self.ax.plot(z_plot,y_plot,color)
            
    def plot_all_fields(self):
        colors = ['r','g','b','m','c','y']
        for idy, Hy in enumerate(self.Hy_all):
            self.plot_rays(Hy, colors[idy%6])
        plt.legend(loc='best')
    
    def plot_optical_axis(self):
        self.ax.plot([min(self.t),max(self.t)],[0,0],'k')
    
    def show(self):
        
        self.init_plot()
        self.surface_vertices()
        self.set_axes()
        self.plot_optical_axis()
        self.draw_all_surfaces()
        self.draw_all_edges()
        self.plot_all_fields()

# **********************************************************************************************
# ==============================================================================================
# **********************************************************************************************
        
# TODO - complete
class plot3D(object):
    
    def __init__(self, lens, **options):
        self.__dict__.update(options)
        self.lens = lens
        
        if not hasattr(self,'num_pupil_rays'):
            self.num_pupil_rays = 5
            
        if not hasattr(self,'num_sag_pts'):
            self.num_sag_pts = 50
    
    def init_plot(self):
        '''initialize plot and define variables'''
        fig = plt.figure()
        self.ax = fig.gca(projection='3d')
        
        self.N = self.lens.num_surfaces
        self.SA = self.lens.SA
        self.C = self.lens.C
        self.n = self.lens.n(self.lens.prim_wave)
        self.F = self.lens.y_field_list()
        
        self.y_list = self.lens.y_field_list()
        self.x_list = self.lens.x_field_list()
        
        if max(self.y_list) is 0:
            self.Hy_all = [0 for y in self.y_list]
        else:
            self.Hy_all = [y/max(self.y_list) for y in self.y_list]
        
        if max(self.x_list) is 0:
            self.Hx_all = [0 for x in self.x_list]
        else:
            self.Hx_all = [x/max(self.x_list) for x in self.x_list]
        
        self.sag_max = np.zeros(self.N-1)
        self.sag_min = np.zeros(self.N-1)
        
    def surface_vertices(self):
        '''find vertices of all surfaces'''
        self.t = np.zeros(self.N)
        self.t[1:] = self.lens.t[:-1]
        if self.t[1]>= 1e10:
            self.t[1] = 10
        self.t = np.cumsum(self.t)
            
    def set_axes(self):
        '''set axes limits'''
        self.ax.set_xlim([min(self.t), max(self.t)])
        self.ax.set_ylim([-1.5*max(self.SA[1:]), 1.5*max(self.SA[1:])])
        
    def sag(self, C, SA, k, **kwargs):
        '''find sag of a surface'''
        if 'A' in kwargs:
            A = kwargs['A']
        else:
            A = [0]
        x = np.linspace(-SA,SA,self.num_sag_pts)
        X,Y = np.meshgrid(x,x)
        R = X**2 + Y**2
        Z = C*R**2/(1 + np.sqrt(1 - R**2*C**2*(k+1))) + \
            sum([(R**2)**(m+1)*a for m,a in enumerate(A)])
        return X,Y,Z
    
    def surface(self, surf):
        '''return 3D x/y/z points for a surface'''
        if self.lens.surf_type[surf] == 'standard':
            X,Y,Z = self.sag(self.C[surf], self.SA[surf], k=0)
        elif self.lens.surf_type[surf] == 'aspheric':
            X,Y,Z = self.sag(C=self.C[surf], SA=self.SA[surf], 
                            k=self.lens.surface_list[surf].conic, A=self.lens.surface_list[surf].A)
        self.sag_max[surf-1] = np.nanmax(Z)
        self.sag_min[surf-1] = np.nanmin(Z)
        self.ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                             linewidth=0, antialiased=False)
        
    def surface_all(self):
        '''plot al surfaces of the system'''
        for surf in range(1,self.N):
            self.surface(surf)
    
    def edge(self, surf):
        '''plot an edge in 3D for a given surface'''
        pass
    
    def edge_all(self):
        '''plot all edges for all surfaces'''
        pass
    
    def trace_ray(self, Hx, Hy, Px, Py):
        pass
    
    def plot_ray(self, Hx, Hy, Px, Py):
        pass
    
    def plot_field(self, Hx, Hy):
        pass
    
    def plot_fields_all(self):
        pass
    
    def show(self):
        self.init_plot()
        self.surface_vertices()
        self.set_axes()
#        self.surface_all()
        self.surface(1)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        
        plt.axis('image')
        plt.show()
        
    
if __name__ == '__main__':
    
    import sample_lenses
    
    system = sample_lenses.aspheric_singlet()
    
    plotter = plot2D(system, num_pupil_rays=10)
    plotter.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    