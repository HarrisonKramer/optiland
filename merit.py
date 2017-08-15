# -*- coding: utf-8 -*-

from operand import operand
from variable import variable
import numpy as np

class merit(object):
    
    def __init__(self):
        self.num_ops = 0
        self.num_vars = 0
        self.ops = []
        self.vars = []
    
    def add_operand(self, lens, type_, target, weight, **kwargs):
        '''add an operand to the merit function'''
        self.ops.append(operand(self.num_ops, lens, type_, target, weight, **kwargs))
        self.num_ops += 1
    
    def add_variable(self, lens, type_, **kwargs):
        '''add a variable to the merit function'''
        self.vars.append(variable(self.num_vars, lens, type_, **kwargs))
        self.num_vars += 1
        
    def fun_array(self):
        '''array of operand target deltas'''
        return np.array([op.fun() for op in self.ops])
    
    def RSS(self):
        '''RSS of current merit function'''
        return np.sqrt(np.sum(np.array(self.fun_array())**2))
        
    def info(self):
        '''print info about merit function'''
        print('Merit Function Information')
        print('  Value: %.6f'%self.RSS())
        print('  Operands: ')
        for op in self.ops:
            op.info()
        print('  Variables: ')
        for var in self.vars:
            var.info()
    
    
    
        