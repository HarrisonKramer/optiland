# -*- coding: utf-8 -*-
"""
merit.py
"""

from lens import lens
import numpy as np

class optimize(object):
    
    def __init__(self):
        self.vars = {}
        self.ops = {}
    
    def add_operand(self, number, type_, target, **kwargs):
        self.ops[type_]= {}
        self.ops[type_]['number'] = number
        self.ops[type_]['target'] = target
        for key, val in kwargs.items():
            self.ops[type_][key] = val
            
    def list_operands(self):
        for op in self.ops:
            print('Operand: ' + str(op))
            for param, val in self.ops[op].items():
                print('\t-' + str(param) + ': ' + str(val))
    
    def add_variable(self, number, type_, **kwargs):
        self.vars[type_] = {}
        self.vars[type_]['number'] = number
        for key, val in kwargs.items():
            self.vars[type_][key] = val
            
    def list_variables(self):
        for var_ in self.vars:
            print('Variable: ' + str(var_))
            for param, val in self.vars[var_].items():
                print('\t-' + str(param) + ': ' + str(val))
                
    def list_delta_targets(self):
        pass
    
    def damped_LS(self, lens, its=100, fac=1.1, tol=1e-6):                
        
        for k in range(its):
            
            # == find initial matrix =========================================================
            mInittmp = np.zeros((len(self.ops),1))
            for idx, type_ in enumerate(self.ops):
                if type_ is 'focal_length':
                    mInittmp[idx] = lens.f2()

            mInit= np.tile(mInittmp,(1,len(self.vars)))
            
            # == initialize final matrix =========================================================
            mFinal = np.zeros_like(mInit)
            
            for m, type_var in enumerate(self.vars):
                        
                # == make a small change to variables =============================================
                if type_var == 'radius':
                    R = lens.radius_array()
                    surf = self.vars[type_var]['surface']
                    Rorig = np.copy(R[surf])
                    lens.set_R(value=fac*R[surf], surface=surf)
                
                # == find final matrix =========================================================
                for n, type_op in enumerate(self.ops):
                    if type_op is 'focal_length':
                        mFinal[n,m] = lens.f2()

                # == return variables to their original state =======================================
                if type_var == 'radius':
                    surf = self.vars[type_var]['surface']
                    lens.set_R(value=Rorig, surface=surf)
                        
            # == find differene matrix =============================================================
            m = mFinal - mInit
            
            # == find desired target deltas =========================================================
            targ = np.zeros(len(self.ops))
            for a, type_ in enumerate(self.ops):
                targ[a] = self.ops[type_]['target']
            delTarg = targ - mInit[:,0]
            
            rms = np.sqrt(np.mean(delTarg)**2)
            print('Iteration %d: RMS = %.8f'%(k+1,rms))
            if rms < tol:
                print('*** Optimization Successful ***')
                break
            c = np.linalg.lstsq(m,delTarg)[0]
                       
            # == limit large magnitude changes to avoid oscillations =================================
            if k == 0:
                g = 0.3
            elif k == 1:
                g = 0.65
            else:
                g = 1
            
            # == make changes to variables =========================================================
            for b, type_var in enumerate(self.vars):
                if type_var == 'radius':
                    surf = self.vars['radius']['surface']
                    R_old = lens.radius_array()[surf]
                    R_new = R_old + g*c[a]/(fac-1)
                    lens.set_R(value=R_new,surface=surf)
        if k == its-1:
            print('Optimization failed for specified tolerance.')
    
if __name__ == '__main__':
    
    # create lens
    singlet = lens(name='Singlet')

    # create surfaces    
    singlet.add_surface(number=0,thickness=25,comment='object')
    singlet.add_surface(number=1,radius=100,index=1.517,thickness=5,stop=True)
    singlet.add_surface(number=2,radius=1000,thickness=47.48)
    singlet.add_surface(number=3,comment='image')
    
    # add fields
    singlet.field_type = 'angle'
    singlet.add_field(x=0,y=0)
    singlet.add_field(x=0,y=7)
    singlet.add_field(x=0,y=10)
    
    # add aeperture type and value
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    # set SA for zero vignetting
    singlet.set_SA()
    
    optimizer = optimize()
    
    optimizer.add_variable(number=0,type_='radius',surface=2)
    optimizer.add_operand(number=0,type_='focal_length',target=511)
    
    optimizer.list_operands()
    optimizer.list_variables()
    
    oldEFL = singlet.f2()
    oldR = singlet.surface_list[2].radius
    optimizer.damped_LS(singlet,its=1000)
    newEFL = singlet.f2()
    newR = singlet.surface_list[2].radius
    
    print('Old: f=%.4f, R=%.4f'%(oldEFL,oldR))
    print('New: f=%.4f, R=%.4f'%(newEFL,newR))
    
    
    