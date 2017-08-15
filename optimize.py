# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 13:22:23 2017

@author: kramer
"""
import numpy as np
from scipy.optimize import minimize

def optimize(merit, maxiter=1000, disp=True):
    
    x0 = [var.value for var in merit.vars]
    bnds = tuple([var.bounds for var in merit.vars])
    
    def fun(x):
        for idvar, var in enumerate(merit.vars):
            var.update(x[idvar])
        funs = np.array([op.fun() for op in merit.ops])
        return np.sum(funs**2)
        
    options = {'maxiter':maxiter,
               'disp':disp}
        
    result = minimize(fun, x0, bounds=bnds, options=options)
    return result, x0
    
def deoptimize(merit, x0):
        for idvar, var in enumerate(merit.vars):
            var.update(x0[idvar])
    
if __name__ == '__main__':
    
    import sample_lenses
    from merit import merit
    from visualize import plot2D
    
    system = sample_lenses.Edmund_49_847()
    
    system.surface_list[1].radius = 100
    system.surface_list[2].radius = -100
    system.surface_list[2].thickness = 95.7828
    
    new_merit = merit()
    new_merit.add_operand(system, type_='f2', target=100, weight=1) # effl
#    new_merit.add_operand(system, type_='radius_greater_than', target=80, surf_num=1, weight=1)
#    new_merit.add_operand(system, type_='real_y', target=24.42, Hx=0, Hy=1, Px=0, Py=0, surf_num=3, wave=0, weight=0)
    new_merit.add_operand(system, type_='seidel', seidel_number=1, target=0, weight=0.1)
#    new_merit.add_operand(system, type_='OPD', Px=0, Py=1, Hx=0, Hy=0, wave=0, target=0, weight=1)
    
    new_merit.add_variable(system, type_='radius', surf_num=1)
    new_merit.add_variable(system, type_='radius', surf_num=2)
#    new_merit.add_variable(system, type_='thickness', surf_num=1)
    
    plot2D(system).show()
    result, x0 = optimize(new_merit)
    new_merit.info()
    
    system.update_paraxial()
    plot2D(system).show()