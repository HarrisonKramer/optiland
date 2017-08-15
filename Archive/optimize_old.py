# -*- coding: utf-8 -*-

import numpy as np

def DLS(lens, merit, its=100, fac=1.1, tol=1e-6, delta_tol=1e-6):
    '''Damped Least-Squares of a lens system
    
    Inputs:
        lens - an instance of the lens class. Defines the lens to be optimized.
        merit - an instance of the merit class. Defines the merit function for
                the optimization.
        its - (optional) number of iterations to perform. Default is 100
        fac - (optional) the factor by which to modify the lens variables to obtain
              the sensitivity matrix (jacobian). Default is 1.1.
        tol - (optional) the merit function value at which optimization should end
              prematurely. Default is 1e-6.
        delta_tol - (optional) the merit function delta between two optimization 
                    consecutive iterations that will signal the optimization to halt.
                    Default is 1e-6.
                
    Outputs:
        merit_value: the final merit function value of the system
    
    '''
    
    old_RSS = 1e10
    
    for it in range(its):
        # == initial jacobian ============================================================
        m_init = np.tile(np.array(merit.op_value()).T, (1, len(merit.vars)))
        
        # == initialize final matrix =====================================================
        m_final = np.zeros_like(m_init)
        
        for idv, var in enumerate(merit.vars):
            # == make a small change to variables ========================================
            var.update(var.value() * fac)
            #print(var.value() * fac)
            
            # == find column of jacobian =================================================
            m_final[:,idv] = np.array(merit.op_value())
            
            # == return variables to their original state ================================
            var.update(var.value() / fac)
            
        # == find differene matrix =======================================================
        m = m_final - m_init
        
        # == find desired target deltas ==================================================
        del_targ = np.array(merit.tar_value()) - m_init[:,0]
        
        # == perform least-squares =======================================================
        c = np.linalg.lstsq(m,del_targ)[0]
        
        # == limit large magnitude changes to avoid oscillations =========================
        if it == 0:
                g = 0.3
        elif it == 1:
            g = 0.65
        else:
            g = 1
            
        # == make changes to variables ===================================================
        for idv, var in enumerate(merit.vars):
            var.update(var.value() + g*c[idv]/(fac-1))
        
        new_RSS = np.sqrt(del_targ**2)
        if (new_RSS < tol) or (old_RSS-new_RSS)<delta_tol:
            break
        
        if it == its-1:
            print('Optimization failed for specified tolerance.')
            
    return merit.RSS(), it
        
if __name__ == '__main__':
    
    from lens import lens
    from merit import merit
    
    # create lens
    singlet = lens(name='Singlet')

    # create surfaces    
    singlet.add_surface(number=0,thickness=25,comment='object')
    singlet.add_surface(number=1,radius=100,material='SF6',thickness=5,stop=True)
    singlet.add_surface(number=2,radius=-100,thickness=47.48)
    singlet.add_surface(number=3,comment='image')
    
    # add fields
    singlet.field_type = 'angle'
    singlet.add_field(number=0,x=0,y=0)
    singlet.add_field(number=1,x=0,y=7)
    singlet.add_field(number=2,x=0,y=10)
    
    # add aeperture type and value
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    # add wavelengths
    singlet.add_wavelength(number=0,value=450)
    singlet.add_wavelength(number=1,value=550, primary=True)
    singlet.add_wavelength(number=2,value=650)
    
    # set SA for zero vignetting
    singlet.set_SA()
    
    merit_fxn = merit()
    
    merit_fxn.add_operand(lens=singlet, type_='seidel', target=0, weight=1.0, seidel_number=0)
#    merit_fxn.add_operand(lens=singlet, type_='f2', target=100, weight=1)
    merit_fxn.add_variable(lens=singlet, type_='radius', number=1)
    merit_fxn.add_variable(lens=singlet, type_='radius', number=2)
    
    print('S1: ' + str(singlet.seidels()[0]))
    print(singlet.R)
    print(DLS(singlet, merit_fxn, its=300, fac=1.14))
    print('S1: ' + str(singlet.seidels()[0]))
    print(singlet.R)   
                
    
    
    
    
    
    