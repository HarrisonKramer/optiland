# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class distribution(object):
    
    def __init__(self,type_='hexapolar', num_pts=100):
        self.num_pts = num_pts
        self.type = type_

        if type_ == 'hexapolar':
            self.hexapolar()
        elif type_ == 'rectangular':
            self.rectangular()
        elif type_ == 'square':
            self.square()
        elif type_ == 'random':
            self.random()
        elif type_ == 'lineX':
            self.lineX()
        elif type_ == 'lineY':
            self.lineY()
        else:
            raise ValueError('Invalid grid distribution type specified.')
            
    @property
    def points(self):
        return zip(self.x,self.y)
    
    def hexapolar(self):
        num_rings = int(np.ceil( -1/2 + np.sqrt(9 - 12*(1-self.num_pts))/6 )) # from arithmetic sum
        r = np.linspace(0,2/np.sqrt(3),num_rings)
        hex_theta = np.linspace(0,5*np.pi/3,6)
        x, y = np.array([0]), np.array([0])
        for ring in range(1, num_rings-1):
            for side in range(6):
                x2 = r[ring]*np.cos(hex_theta[(side+1)%6])
                x1 = r[ring]*np.cos(hex_theta[side])
                y2 = r[ring]*np.sin(hex_theta[(side+1)%6])
                y1 = r[ring]*np.sin(hex_theta[side])
                x_new = x1 + np.arange(ring+1)/ring*(x2 - x1)
                y_new = y1 + np.arange(ring+1)/ring*(y2 - y1)
                x = np.concatenate((x,x_new[1:]))
                y = np.concatenate((y,y_new[1:]))
        
        R = np.sqrt(x**2+y**2)
        self.x, self.y = x[R<1], y[R<1]
    
    def rectangular(self):
        num = np.sqrt(4*self.num_pts / np.pi) # number of points along one axis across EP
        xi = np.linspace(-1,1,np.ceil(num))
        yi = np.linspace(-1,1,np.ceil(num))
        x, y = np.meshgrid(xi, yi)
        r = np.sqrt(x**2+y**2)
        self.x, self.y = x[r<1], y[r<1]
        
    def square(self):
        '''square distribution'''
        num = int(np.sqrt(self.num_pts))
        SQR2 = np.sqrt(2)/2
        xi = np.linspace(-SQR2,SQR2,num)
        yi = np.linspace(-SQR2,SQR2,num)
        self.x, self.y = np.meshgrid(xi, yi)
    
    def random(self):
        num = int(np.sqrt(self.num_pts)*4/np.pi)
        x = 2*np.random.rand(num,num)-1
        y = 2*np.random.rand(num,num)-1
        r = np.sqrt(x**2+y**2)
        self.x, self.y = x[r<1], y[r<1]
        
    def concentric(self):
        pass
            
        
    def lineX(self):
        self.x = np.linspace(-1,1,self.num_pts)
        self.y = np.zeros_like(self.x)
        
    def lineY(self):
        self.y = np.linspace(-1,1,self.num_pts)
        self.x = np.zeros_like(self.y)
    
    def show(self):
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y,'k*')
        t = np.linspace(0,2*np.pi,150)
        x, y = np.cos(t), np.sin(t)
        ax.plot(x,y,'r')
        plt.title('%s Pupil Distribution'%self.type.title())
        ax.set_xlabel('Normalized Pupil Coordiante X')
        ax.set_ylabel('Normalized Pupil Coordinate Y')
        plt.axis('equal')
    
if __name__ == '__main__':
    
    x = distribution('rectangular')
    x.show()
    print(len(x.x))
    
    
    