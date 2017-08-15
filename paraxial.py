# -*- coding: utf-8 -*-
"""
paraxial.py
"""

from trace import paraxial_trace
import numpy as np

class paraxial(object):
    
    def __init__(self, lens):
        self.lens = lens
        
    def surface_powers(self):
        return (self.lens.n(self.lens.prim_wave)[1:]-self.lens.n(self.lens.prim_wave)[:-1])/self.lens.R[1:]
    
    def f1(self):
        '''front focal length'''
        y, u = paraxial_trace(self.lens,y0=1,u0=0,wave=self.lens.prim_wave,reverse=True,last_transfer=False)
        return y[-1]/u[0]
    
    def f2(self, wave='primary'):
        '''focal length'''
        if wave == 'primary':
            wave = self.lens.prim_wave
        y, u = paraxial_trace(self.lens,y0=1,u0=0,wave=wave,first_transfer=False)
        return -y[0]/u[-1]
        
    def F1(self):
        '''front focus (ffl)'''
        y, u = paraxial_trace(self.lens,y0=1,u0=0,wave=self.lens.prim_wave,reverse=True,last_transfer=False)
        return y[1]/u[0]
        
    def F2(self):
        '''back focus (bfl)'''
        y, u = paraxial_trace(self.lens,y0=1,u0=0,wave=self.lens.prim_wave,first_transfer=False)
        return -y[-2]/u[-1]
        
    def P1(self):
        '''first principal point'''
        return self.f1() - self.F1()
        
    def P2(self):
        '''second principal point'''
        return self.f2() - self.F2()
        
    def N1(self):
        '''first nodal point'''
        return self.P2()*self.F2()/self.F1()
        
    def N2(self):
        '''second nodal point'''
        return self.P1()*self.F1()/self.F2()    
            
    # ===================================================================================================================
    def set_SA(self):
        '''set SA for zero vignetting'''
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,y0=ya0,wave=self.lens.prim_wave,u0=ua0,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,y0=yb0,wave=self.lens.prim_wave,u0=ub0,first_transfer=False)
        for k in range(1,len(self.lens.surface_list)):
            self.lens.surface_list[k].semi_aperture = np.abs(ya[k-1]) + np.abs(yb[k-1])
        
    # ===================================================================================================================    
    def EPD(self):
        '''entrance pupil diameter'''
        if self.lens.stop_surf() == 1:
            return 2*self.lens.SA[1]
            
        else:
            m = 1
            lp = self.lens.t[self.lens.stop_surf()-1]
            for k in range(self.lens.stop_surf()-1,0,-1):
                l = self.lens.n(self.lens.prim_wave)[k-1]*lp/(self.lens.n(self.lens.prim_wave)[k] - lp*self.surface_powers()[k])
                m *= self.lens.n(self.lens.prim_wave)[k-1]*lp/(self.lens.n(self.lens.prim_wave)[k]*l)
                if k is not 1:
                    lp = self.lens.n(self.lens.prim_wave)[k-1] - l
                        
            return self.lens.SA[self.lens.stop_surf()]*m*2
    
    # ===================================================================================================================            
    def EPL(self):
        '''entrance pupil location (w.r.t first surface)'''
        if self.lens.stop_surf() == 1:
            return 0
            
        else:
            stop = self.lens.stop_surf()
            n = self.lens.n(self.lens.prim_wave)
            R = self.lens.R
            t = self.lens.t
            sp = t[stop-1]
            for k in range(stop-1,0,-1):
                s = n[k-1]*R[k]*sp/(sp*(n[k]-n[k-1]) - n[k]*R[k])
                if k is not 1:
                    sp = t[k-1] - s
                    
            return s
    
    # ===================================================================================================================    
    def XPD(self):
        '''exit pupil diameter'''        
        if self.lens.stop_surf() == self.lens.num_surfaces - 1:
            return self.lens.SA[self.lens.stop_surf()]
            
        else:
            m = 1
            l = self.lens.t[self.lens.stop_surf()]
            for k in range(self.lens.stop_surf(),self.lens.num_surfaces-2):
                lp = self.lens.n(self.lens.prim_wave)[k+1]*l/(self.surface_powers()[k+1]*l + self.lens.n(self.lens.prim_wave)[k])
                m *= self.lens.n(self.lens.prim_wave)[k]*lp/(self.lens.n(self.lens.prim_wave)[k+1]*l)
                if k is not self.lens.num_surfaces-1:
                    l = self.lens.t[k+1] - lp
                    
            return self.lens.SA[self.lens.stop_surf()]*m*2
    
    # ===================================================================================================================    
    def XPL(self):
        '''exit pupil location (w.r.t. last surface)'''
        if self.lens.stop_surf() == self.lens.num_surfaces - 1:
            return 0
            
        else:
            m = 1
            l = self.lens.t[self.lens.stop_surf()]
            for k in range(self.lens.stop_surf(),self.lens.num_surfaces-2):
                lp = self.lens.n(self.lens.prim_wave)[k+1]*l/(self.surface_powers()[k+1]*l + self.lens.n(self.lens.prim_wave)[k])
                m *= self.lens.n(self.lens.prim_wave)[k]*lp/(self.lens.n(self.lens.prim_wave)[k+1]*l)
                if k is not self.lens.num_surfaces-1:
                    l = self.lens.t[k+1] - lp
                    
            return l
    
    # ===================================================================================================================            
    def FNO(self):
        '''image-space F/#'''
        return self.f2()/self.EPD()
        
    # ===================================================================================================================  
    def pupil_mag(self):
        '''pupil magnification'''
        return self.XPD()/self.EPD()
        
    # ===================================================================================================================
    def objectNA(self):
        '''object-space numerical aperture'''
        y,u = paraxial_trace(self.lens,y0=0,u0=0.1,wave=self.lens.prim_wave)
        ua0 = 0.1*self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()]
        return self.lens.n(self.lens.prim_wave)[0]*np.sin(ua0)
        
    # ===================================================================================================================
    def imageNA(self):
        '''image-space numerical aperture'''
        return self.lens.n(self.lens.prim_wave)[0]*np.sin(np.arctan(self.EPD()/(2*self.f2())))
    
    # ===================================================================================================================
    def object_cone_angle(self):
        '''object-space cone angle'''
        y,u = paraxial_trace(self.lens,y0=0,u0=0.1,wave=self.lens.prim_wave)
        ua0 = 0.1*self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()]
        return 2*np.rad2deg(ua0)
        
    def marginal_ray(self):
        '''marginal ray of system, returns y and u at first surface'''
        
        if self.lens.aperture_type is 'EPD':
            EPD_c = self.EPD()
            if self.lens.surface_list[0].thickness >= 1e10:
                y,u = paraxial_trace(self.lens,y0=1,u0=0,wave=self.lens.prim_wave,first_transfer=False)
                ya0 = np.abs(self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()-1])
                return ya0*self.lens.aperture_value/EPD_c, 0
            else:
                y,u = paraxial_trace(self.lens,y0=0,u0=0.1,wave=self.lens.prim_wave)
                ua0 = 0.1*self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()]
                ua = ua0*self.lens.aperture_value/EPD_c
                return ua*self.lens.surface_list[0].thickness, ua
            
        elif self.lens.aperture_type is 'imageNA':
            NA_c = self.imageNA()
            y,u = paraxial_trace(self.lens,y0=0,u0=0.1,wave=self.lens.prim_wave)
            ua0 = 0.1*self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()]
            ua = ua0*self.lens.aperture_value/NA_c
            return ua*self.lens.surface_list[0].thickness, ua
        
        elif self.lens.aperture_type is 'imageFNO':
            FNO_c = self.FNO()
            y,u = paraxial_trace(self.lens,y0=1,u0=0,wave=self.lens.prim_wave,first_transfer=False)
            ya0 = np.abs(self.lens.SA[self.lens.stop_surf()]/y[self.lens.stop_surf()-1])
            return ya0*FNO_c/self.lens.aperture_value, 0
            
        elif self.lens.aperture_type == 'objectNA':
            ua = np.asin(self.lens.aperture_value/self.lens.surface_list[0].index)
            return ua*self.lens.surface_list[0].thickness, ua
        
        elif self.lens.aperture_type is 'object_cone_angle':
            ua = np.deg2rad(self.lens.aperture_value)/2
            return ua*self.lens.surface_list[0].thickness, ua
        
        else:
            raise ValueError('Invalid aperture type specified.')
    
    def chief_ray(self):
        '''find chief ray of system, return y and u at first surface'''
        
        y,u = paraxial_trace(self.lens,y0=0,u0=0.1,wave=self.lens.prim_wave,reverse=True,last=self.lens.stop_surf())
        
        # find max y field
        max_y_field = 0
        for field in self.lens.fields_list:
            if field[1] > max_y_field:
                max_y_field = field[1]
        
        if self.lens.field_type == 'object_height':
            yn,un = paraxial_trace(self.lens,y0=0,u0=0.1*max_y_field/y[0],wave=self.lens.prim_wave,reverse=True,last=self.lens.stop_surf())
        
        elif self.lens.field_type == 'angle':
            yn,un = paraxial_trace(self.lens,y0=0,u0=0.1*np.deg2rad(max_y_field)/u[0],wave=self.lens.prim_wave,reverse=True,last=self.lens.stop_surf())
            
        else:
            raise ValueError('Invalid field type specified.')
    
        return yn[1], un[0]
    
    # ===================================================================================================================        
    def ABCD(self):
        '''find ABCD matrix of self.lens system'''
        ABCD = np.eye(2)
        for k in range(self.lens.num_surfaces-2,0,-1):
            ABCD = np.dot(ABCD,np.array([[1, 0],[-self.surface_powers()[k], 1]]))
            if k is not 1:
                ABCD = np.dot(ABCD,np.array([[1, self.lens.t[k-1]/self.lens.n(self.lens.prim_wave)[k-1]],[0, 1]]))
            
        return ABCD
        
    # ===================================================================================================================
    def angular_mag(self):
        '''angular magnification of lens'''
        yb0,ub0 = self.chief_ray()
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        return ub[-1]/ub[0]
    
    # ===================================================================================================================
    def m(self):
        '''magnification'''
        ya0,ua0 = self.marginal_ray()
        ya,ua = paraxial_trace(self.lens,y0=ya0,u0=ua0,wave=self.lens.prim_wave,first_transfer=False)
        return self.lens.n(self.lens.prim_wave)[0]*ua[0]/(self.lens.n(self.lens.prim_wave)[-1]*ua[-1])
    
    # ===================================================================================================================
    def Inv(self):
        '''invariant'''
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        return yb0*self.lens.n(self.lens.prim_wave)[0]*ua0 - ya0*self.lens.n(self.lens.prim_wave)[0]*ub0
    
    # ===================================================================================================================
    def total_track(self):
        '''total track'''
        return np.sum(self.lens.t[1:-1])
        
    # ===================================================================================================================
    def image_distance_solve(self):
        self.lens.set_t(self.F2(), self.lens.num_surfaces-2)
        return self
        
    # ===================================================================================================================
    def update_paraxial(self):
        self.set_SA()
        if self.lens.image_solve:
            self.image_distance_solve()
        
# ===================================================================================================================
    def third_order_abs(self):
        '''third-order aberrations'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = self.Inv()/(n[-1]*ua[-1])
        dn = (n-1)/self.lens.v
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TSC = np.zeros(N-2)
        SC = np.zeros(N-2)
        CC = np.zeros(N-2)
        TAC = np.zeros(N-2)
        AC = np.zeros(N-2)
        TPC = np.zeros(N-2)
        PC = np.zeros(N-2)
        DC = np.zeros(N-2)
        TAchC = np.zeros(N-2)
        LachC = np.zeros(N-2)
        TchC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            #TODO - update for aspherics
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp + W*ya**4*hp
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp + W*ya**3*yb*hp
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp + W*ya**2*yb**2*hp
                TPC[k-1] = (n[k]-n[k-1])*Ce*hp*Inv/(2*n[k]*n[k-1])
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2)) + W*ya*yb**3*hp
                TAchC[k-1] = -ya[k-1]*i[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
                TchC[k-1] = -ya[k-1]*ip[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp
                TPC[k-1] = (n[k]-n[k-1])*C[k]*hp*Inv/(2*n[k]*n[k-1])
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2))
                TAchC[k-1] = -ya[k-1]*i[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
                TchC[k-1] = -ya[k-1]*ip[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
                
            SC[k-1] = -TSC[k-1]/ua[-1]
            AC[k-1] = -TAC[k-1]/ua[-1]
            PC[k-1] = -TPC[k-1]/ua[-1]
            LachC[k-1] = -TAchC[k-1]/ua[-1]
    
        S = np.array([-sum(TSC)*n[-1]*ua[-1]*2,-sum(CC)*n[-1]*ua[-1]*2,
                      -sum(TAC)*n[-1]*ua[-1]*2,-sum(TPC)*n[-1]*ua[-1]*2,
                      -sum(DC)*n[-1]*ua[-1]*2])
    
        return TSC,SC,CC,TAC,AC,TPC,PC,DC,TAchC,LachC,TchC,S
        
    # ===================================================================================================================
    def seidels(self):
        '''seidel coefficients'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TSC = np.zeros(N-2)
        CC = np.zeros(N-2)
        TAC = np.zeros(N-2)
        TPC = np.zeros(N-2)
        DC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp + W*ya**4*hp
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp + W*ya**3*yb*hp
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp + W*ya**2*yb**2*hp
                TPC[k-1] = (n[k]-n[k-1])*Ce*hp*Inv/(2*n[k]*n[k-1])
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2)) + W*ya*yb**3*hp
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp
                TPC[k-1] = (n[k]-n[k-1])*C[k]*hp*Inv/(2*n[k]*n[k-1])
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2))
    
        S = np.array([-sum(TSC)*n[-1]*ua[-1]*2,-sum(CC)*n[-1]*ua[-1]*2,
                      -sum(TAC)*n[-1]*ua[-1]*2,-sum(TPC)*n[-1]*ua[-1]*2,
                      -sum(DC)*n[-1]*ua[-1]*2])
    
        return S
    
    # ===================================================================================================================
    def TSC(self):
        '''3rd order transverse spherical aberration'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TSC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp + W*ya**4*hp
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TSC[k-1] = B[k-1]*i[k-1]**2*hp
    
        return TSC
    
    # ===================================================================================================================
    def CC(self):
        '''3rd order sagittal coma'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        CC = np.zeros(N-2)
    
        for k in range(1,N-1,1):
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp + W*ya**3*yb*hp
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                CC[k-1] = B[k-1]*i[k-1]*ip[k-1]*hp
    
        return CC
    
    # ===================================================================================================================
    def CC3(self):
        '''3rd order tangential coma'''
        return 3*self.CC()
    
    # ===================================================================================================================
    def TAC(self):
        '''3rd order transverse astigmatism'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TAC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
        
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp + W*ya**2*yb**2*hp
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TAC[k-1] = B[k-1]*ip[k-1]**2*hp
    
        return TAC
    
    # ===================================================================================================================
    def TPC(self):
        '''3rd order transverse Petzval contribution'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TPC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TPC[k-1] = (n[k]-n[k-1])*Ce*hp*Inv/(2*n[k]*n[k-1])
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TPC[k-1] = (n[k]-n[k-1])*C[k]*hp*Inv/(2*n[k]*n[k-1])
    
        return TPC
    
    # ===================================================================================================================
    def DC(self):
        '''3rd order distortion'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        hp = Inv/(n[-1]*ua[-1])
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        DC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                K = A[k][1] - A[k][0]/4*(4*A[k][0]**2 + 6*C[k]*A[k][0] +  3*C[k]**2)
                W = 4*K*(n[k]-n[k-1])/Inv
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2)) + W*ya*yb**3*hp
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                DC[k-1] = hp*(Bp[k-1]*i[k-1]*ip[k-1] + 0.5*(ub[k]**2 - ub[k-1]**2))
    
        return DC
    
    # ===================================================================================================================
    def TAchC(self):
        '''paraxial transverse axial chromatic aberration'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        dn = (self.lens.n(self.lens.prim_wave)-1)/lens.v
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TAchC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TAchC[k-1] = -ya[k-1]*i[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TAchC[k-1] = -ya[k-1]*i[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
    
        return TAchC
    
    # ===================================================================================================================
    def TchC(self):
        '''paraxial lateral chromatic aberration'''
        Inv = self.Inv()
        n = self.lens.n(self.lens.prim_wave)
        N = self.lens.num_surfaces
        C = self.lens.C
        ya0,ua0 = self.marginal_ray()
        yb0,ub0 = self.chief_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        yb,ub = paraxial_trace(self.lens,yb0,ub0,wave=self.lens.prim_wave,first_transfer=False)
        dn = (self.lens.n(self.lens.prim_wave)-1)/lens.v
        
        i = np.zeros(N-2)
        ip = np.zeros(N-2)
        B = np.zeros(N-2)
        Bp = np.zeros(N-2)
        TchC = np.zeros(N-2)
        
        for k in range(1,N-1,1):
            
            if self.lens.surf_type[k] == 'aspheric':
                Ce = C[k] + 2*A[k][0]
                i[k-1] = Ce*ya[k-1] + ua[k-1]
                ip[k-1] = Ce*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TchC[k-1] = -ya[k-1]*ip[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
            else:
                i[k-1] = C[k]*ya[k-1] + ua[k-1]
                ip[k-1] = C[k]*yb[k-1] + ub[k-1]
                B[k-1] = n[k-1]*(n[k]-n[k-1])*ya[k-1]*(ua[k]+i[k-1])/(2*n[k]*Inv)
                Bp[k-1] = n[k-1]*(n[k]-n[k-1])*yb[k-1]*(ub[k]+ip[k-1])/(2*n[k]*Inv)
                TchC[k-1] = -ya[k-1]*ip[k-1]/(n[-1]*ua[-1])*(dn[k-1] - n[k-1]/n[k]*dn[k])
    
        return TchC
        
    # ===================================================================================================================        
    def SC(self):
        '''3rd order longitudinal spherical aberration'''
        ya0,ua0 = self.marginal_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        return self.TSC()/ua[-1]
    
    # ===================================================================================================================
    def AC(self):
        '''3rd order longitudinal astigmatism'''
        ya0,ua0 = self.marginal_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        return self.TAC()/ua[-1]
    
    # ===================================================================================================================
    def PC(self):
        '''3rd order longitudinal petzval contribution'''
        ya0,ua0 = self.marginal_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        return self.TPC()/ua[-1]
    
    # ===================================================================================================================
    def LchC(self):
        '''3rd order longitudinal axial color'''
        ya0,ua0 = self.marginal_ray()
        ya,ua = paraxial_trace(self.lens,ya0,ua0,wave=self.lens.prim_wave,first_transfer=False)
        return self.TAchC()/ua[-1]

if __name__ == '__main__':
    
    from lens import lens
    
    # create self.lens
    singlet = lens(name='Singlet')

    # create surfaces    
    singlet.add_surface(number=0,thickness=25,comment='object')
    singlet.add_surface(number=1,radius=26.2467,material='SF6',thickness=5,stop=True)
    singlet.add_surface(number=2,thickness=47.48)
    singlet.add_surface(number=3,comment='image')
    
    # add fields
    singlet.field_type = 'angle'
    singlet.add_field(x=0,y=0)
    singlet.add_field(x=0,y=7)
    singlet.add_field(x=0,y=10)
    
    # add aeperture type and value
    singlet.aperture_type = 'EPD'
    singlet.aperture_value = 10
    
    singlet.add_wavelength(550)
    
    A = paraxial(singlet)
    
    # print various paraxial values
    print('P: ' + str(A.surface_powers()))
    print('a-ray: ' + str(A.marginal_ray()))
    print('b-ray: ' + str(A.chief_ray()))
    print('f1: ' + str(A.f1()))
    print('f2: ' + str(A.f2()))
    print('F1: ' + str(A.F1()))
    print('F2: ' + str(A.F2()))
    print('P1: ' + str(A.P1()))
    print('P2: ' + str(A.P2()))
    print('N1: ' + str(A.N1()))
    print('N2: ' + str(A.N2()))
    print('EPD: ' + str(A.EPD()))
    print('EPL: ' + str(A.EPL()))
    print('XPD: ' + str(A.XPD()))
    print('XPL: ' + str(A.XPL()))
    print('FNO: ' + str(A.FNO()))
    print('Pupil Mag: ' + str(A.pupil_mag()))
    print('Object NA: ' + str(A.objectNA()))
    print('Image NA: ' + str(A.imageNA()))
    print('Object Cone Angle: ' + str(A.object_cone_angle()))
    print('ABCD: ' + str(A.ABCD()))
    print('m: ' + str(A.m()))
    print('Inv: ' + str(A.Inv()))
    print('Total Track: ' + str(A.total_track()))
    
    print('\nSA: ' + str(singlet.SA))
    print('Setting SAs for zero vignetting...')
    A.set_SA()
    print('SA: ' + str(singlet.SA))
    
    third_order_abs = A.third_order_abs()
    abs_list = ['\nTSC','SC','CC','TAC','AC','TPC','PC','DC','TAchC','LachC','TchC','Seidels']
    for k, ab in enumerate(third_order_abs):
        print(abs_list[k] + ': ' + str(ab))


































