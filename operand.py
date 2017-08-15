# -*- coding: utf-8 -*-

from trace import trace
from wavefront import wavefront
from distribution import distribution
from zernike import fit
        
class operand(object):
    
    _inequality_operands = ['thickness_less_than', 'thickness_greater_than',
                                 'radius_less_than', 'radius_greater_than']
    
    def __init__(self, number, lens, type_, target, weight, **kwargs):
        self.__dict__.update(kwargs)
        self.number = number
        self.lens = lens
        self.type_ = type_
        self.target = target
        self.weight = weight
        
    @property
    def value(self):
        '''current value of the operand'''
        
        # === FIRST ORDER ======================================================================
        if self.type_ == 'f1':
            return self.lens.f1()
        elif self.type_ == 'f2':
            return self.lens.f2()
        elif self.type_ == 'F1':
            return self.lens.F2()
        elif self.type_ == 'F2':
            return self.lens.F2()
        elif self.type_ == 'P1':
            return self.lens.P1()
        elif self.type_ == 'P2':
            return self.lens.P2()
        elif self.type_ == 'N1':
            return self.lens.N1()
        elif self.type_ == 'N2':
            return self.lens.N2()
        elif self.type_ == 'FNO':
            return self.lens.FNO()
        elif self.type_ == 'pupil_mag':
            return self.lens.pupil_mag()
        elif self.type_ == 'objectNA':
            return self.lens.objectNA()
        elif self.type_ == 'imageNA':
            return self.lens.imageNA()
        elif self.type_ == 'object_cone_angle':
            return self.lens.object_cone_angle()
        elif self.type_ == 'm':
            return self.lens.m()
        elif self.type_ == 'seidel':
            return self.lens.seidels()[self.seidel_number-1]
        elif self.type_ == 'TSC':
            return self.lens.TSC()[self.surf_num-1]
        elif self.type_ == 'CC':
            return self.lens.CC()[self.surf_num-1]
        elif self.type_ == 'TAC':
            return self.lens.TAC()[self.surf_num-1]
        elif self.type_ == 'TPC':
            return self.lens.TPC()[self.surf_num-1]
        elif self.type_ == 'DC':
            return self.lens.DC()[self.surf_num-1]
        elif self.type_ == 'TAchC':
            return self.lens.TAchC()[self.surf_num-1]
        elif self.type_ == 'TchC':
            return self.lens.TchC()[self.surf_num-1]
        elif self.type_ == 'SC':
            return self.lens.SC()[self.surf_num-1]
        elif self.type_ == 'AC':
            return self.lens.AC()[self.surf_num-1]
        elif self.type_ == 'PC':
            return self.lens.PC()[self.surf_num-1]
        elif self.type_ == 'LchC':
            return self.lens.LchC()[self.surf_num-1]
            
        # === RAYTRACING ======================================================================
        elif self.type_ == 'real_x':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.x[self.surf_num]
        elif self.type_ == 'real_y':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.y[self.surf_num]
        elif self.type_ == 'real_z':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.z[self.surf_num]
        elif self.type_ == 'real_K':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.K[self.surf_num]
        elif self.type_ == 'real_L':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.L[self.surf_num]
        elif self.type_ == 'real_M':
            raytrace = trace(self.lens, self.Hx, self.Hy, self.Px, self.Py, self.wave).run()
            return raytrace.M[self.surf_num]
            
        # TODO - fix this bish
#        elif self.type_ == 'OPD':
#            EPR = self.lens.EPD()/2
#            point = distribution(type_='point', x=EPR*self.Px, y=EPR*self.Py)
#            wvfrnt = wavefront(self.lens, point, Hx=0, Hy=0, wave=0)
#            return wvfrnt.OPD_array()
            
        # === INEQUALITIES ====================================================================
        elif self.type_ == 'radius_less_than':
            R = self.lens.R[self.surf_num]
            if R > self.target:
                return self.lens.R[self.surf_num]
            else:
                return self.target
        elif self.type_ == 'radius_greater_than':
            R = self.lens.R[self.surf_num]
            if R < self.target:
                return self.lens.R[self.surf_num]
            else:
                return self.target
        elif self.type_ == 'thickness_less_than':
            t = self.lens.t[self.surf_num]
            if t > self.target:
                return self.lens.t[self.surf_num]
            else:
                return self.target
        elif self.type_ == 'thickness_greater_than':
            t = self.lens.R[self.surf_num]
            if t < self.target:
                return self.lens.t[self.surf_num]
            else:
                return self.target
                
        # TODO - finish this bish
        # === ZERNIKES ======================================================================
        elif self.type_ == 'zernike_term':
            pass
                
        else:
            raise ValueError('Invalid operand ' + str(self.type_))
            
    def info(self):
        print('\t Number: %d'%self.number)
        print('\t   System: %s'%self.lens.name)
        print('\t   Type: %s'%self.type_)
        print('\t   Weight: %.3f'%self.weight)
        print('\t   Target: %.6f'%self.target)
        print('\t   Value: %.6f'%self.value)
        print('\t   Delta: %.6f'%self.delta())
            
    def delta(self):
        '''delta between target and current value'''
        return (self.value - self.target)
            
    def fun(self):
        '''return objective function value'''
        return self.weight * self.delta()























