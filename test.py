# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 19:55:00 2017

@author: kramer
"""

import analysis
import diffraction
import material
import distribution
import lens
import merit
import operand
import optimize
import paraxial
import sample_lenses
import surface
import trace
import variable
import visualize
import wavefront

# ===================================================================================================
# ****** SYSTEM DEFINITIONS *************************************************************************
# ===================================================================================================

system = sample_lenses.microscope()

# ===================================================================================================
# ****** GRID ***************************************************************************************
# ===================================================================================================
hex_grid = distribution.distribution(type_='hexapolar')
#rect_grid = distribution.distribution(type_='rectangular')
#random_grid = distribution.distribution(type_='random')
#square_grid = distribution.distribution(type_='square')

#hex_grid.show()
#rect_grid.show()
#random_grid.show()
#square_grid.show()

# ===================================================================================================
# ****** ANALYSIS ***********************************************************************************
# ===================================================================================================
analysis.ray_fan(system, Hx=0, Hy=0)
analysis.spot_diagram_standard(system, hex_grid, Hx=0, Hy=0, airy=True)
#analysis.spot_diagram_through_focus(system, hex_grid, Hx=0, Hy=1, total_span=3)
#analysis.distortion(system, type_='f_tan_theta')
#analysis.grid_distortion(system, num_pts=100, type_='f_tan_theta', wave=1)
#analysis.field_curvature(system)
analysis.y_ybar(system, wave=1)
analysis.OPD_fan(system, Hx=0, Hy=1, wave=0)

# ===================================================================================================
# ****** DIFFRACTION ********************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** GLASS **************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** LENS ***************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** MERIT **************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** OPERAND ************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** OPTIMIZE ***********************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** PARAXIAL ***********************************************************************************
# ===================================================================================================
print('P: ' + str(system.surface_powers()))
print('a-ray: ' + str(system.marginal_ray()))
print('b-ray: ' + str(system.chief_ray()))
print('f1: ' + str(system.f1()))
print('f2: ' + str(system.f2()))
print('F1: ' + str(system.F1()))
print('F2: ' + str(system.F2()))
print('P1: ' + str(system.P1()))
print('P2: ' + str(system.P2()))
print('N1: ' + str(system.N1()))
print('N2: ' + str(system.N2()))
print('EPD: ' + str(system.EPD()))
print('EPL: ' + str(system.EPL()))
print('XPD: ' + str(system.XPD()))
print('XPL: ' + str(system.XPL()))
print('FNO: ' + str(system.FNO()))
print('Pupil Mag: ' + str(system.pupil_mag()))
print('Object NA: ' + str(system.objectNA()))
print('Image NA: ' + str(system.imageNA()))
print('Object Cone Angle: ' + str(system.object_cone_angle()))
print('ABCD: ' + str(system.ABCD()))
print('m: ' + str(system.m()))
print('Inv: ' + str(system.Inv()))
print('Total Track: ' + str(system.total_track()))
third_order_abs = system.third_order_abs()
abs_list = ['\nTSC','SC','CC','TAC','AC','TPC','PC','DC','TAchC','LachC','TchC','Seidels']
for k, ab in enumerate(third_order_abs):
    print(abs_list[k] + ': ' + str(ab))
    
# ===================================================================================================
# ****** SPOT ***************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** SURFACE ************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** TRACE **************************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** VARIABLE ***********************************************************************************
# ===================================================================================================

# ===================================================================================================
# ****** VISUALIZE **********************************************************************************
# ===================================================================================================
visualize.plot2D(system).show()

# ===================================================================================================
# ****** WAVEFRONT **********************************************************************************
# ===================================================================================================
















