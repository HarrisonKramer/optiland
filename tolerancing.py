# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:54:05 2017

@author: kramer
"""

from operand import operand
from variable import variable
from merit import merit
from optimize import optimize

# === TOLERANCER =============================================================================
class tolerancer(object):
    
    def __init__(self, lens):
        lens_copy = type('lens', lens.__bases__, dict(lens.__dict__))
        self.lens = lens_copy()
        self.ops = []
        self.vars = []
        self.comps = []
        
        self.num_ops = 0
        self.num_vars = 0
        self.num_comps = 0
    
    def add_operand(self, type_, target, weight, **kwargs):
        '''add an operand to the tolerancer'''
        self.ops.append(operand(self.num_ops, self.lens, type_, target, weight, **kwargs))
        self.nump_ops += 1
        
    def add_variable(self, type_, min_val, max_val, **kwargs):
        '''add a variable to the tolerancer'''
        self.vars.append(variable(self.num_vars, self.lens, type_, min_val, max_val, **kwargs))
        self.num_vars += 1
    
    def add_compensator(self, type_, **kwargs):
        '''add a compensator (variable used in optimization)'''
        self.comps.append(variable(self.num_comps, lens, type_, **kwargs))
        self.num_comps += 1
    
    def _merit(self):
        '''create merit function for optimization with operands and compensators'''
        merit_tol = merit()
        merit_tol.ops = self.ops
        merit_tol.vars = self.comps
        return merit_tol
    
    def _optimize(self, maxiter=1000, disp=False):
        '''optimize the system using compensators and operands'''
        optimize(self._merit, maxiter, disp)
        
    def _sensitivities(self):
        pass
    
    def monte_carlo(self):
        pass
            
    def report(self):
        pass
        
if __name__ == '__main__':
    
    tol = tolerancer()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    