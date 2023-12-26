import numpy as np


class Aberrations:

    def __init__(self, optic):
        self.optic = optic

    def third_order(self):
        """Modern Optical Engineering, Warren Smith - Chapter 6.3"""
        self._precalculations()

        TSC = []
        SC = []
        CC = []
        TAC = []
        AC = []
        TPC = []
        PC = []
        DC = []
        TAchC = []
        LchC = []
        TchC = []

        for k in range(1, self._N-1):
            TSC.append(self._B[k-1] * self._i[k-1]**2 * self._hp)
            CC.append(self._B[k-1] * self._i[k-1] * self._ip[k-1] * self._hp)
            TAC.append(self._B[k-1] * self._ip[k-1]**2 * self._hp)
            TPC.append((self._n[k] - self._n[k-1]) * self._C[k] * self._hp *
                       self._inv / (2*self._n[k] * self._n[k-1]))
            DC.append(self._hp * (self._Bp[k-1] * self._i[k-1] * self._ip[k-1] +
                                  0.5*(self._ub[k]**2 - self._ub[k-1]**2)))
            TAchC.append(-self._ya[k-1] * self._i[k-1] / (self._n[-1] * self._ua[-1]) *
                         (self._dn[k-1] - self._n[k-1] / self._n[k] * self._dn[k]))
            TchC.append(-self._ya[k-1] * self._ip[k-1] / (self._n[-1] * self._ua[-1]) *
                        (self._dn[k-1] - self._n[k-1] / self._n[k] * self._dn[k]))

            SC.append(-TSC[-1] / self._ua[-1])
            AC.append(-TAC[-1] / self._ua[-1])
            PC.append(-TPC[-1] / self._ua[-1])
            LchC.append(-TAchC[-1] / self._ua[-1])

        S = np.array([-sum(TSC) * self._n[-1] * self._ua[-1]*2,
                      -sum(CC) * self._n[-1] * self._ua[-1]*2,
                      -sum(TAC) * self._n[-1] * self._ua[-1]*2,
                      -sum(TPC) * self._n[-1] * self._ua[-1]*2,
                      -sum(DC) * self._n[-1] * self._ua[-1]*2])

        return TSC, SC, CC, CC*3, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S

    def seidels(self):
        self._precalculations()

        TSC = []
        CC = []
        TAC = []
        TPC = []
        DC = []

        for k in range(1, self._N-1):
            TSC.append(self._B[k-1] * self._i[k-1]**2 * self._hp)
            CC.append(self._B[k-1] * self._i[k-1] * self._ip[k-1] * self._hp)
            TAC.append(self._B[k-1] * self._ip[k-1]**2 * self._hp)
            TPC.append((self._n[k] - self._n[k-1]) * self._C[k] * self._hp *
                       self._inv / (2*self._n[k] * self._n[k-1]))
            DC.append(self._hp * (self._Bp[k-1] * self._i[k-1] * self._ip[k-1] +
                                  0.5*(self._ub[k]**2 - self._ub[k-1]**2)))

        S = np.array([-sum(TSC) * self._n[-1] * self._ua[-1]*2,
                      -sum(CC) * self._n[-1] * self._ua[-1]*2,
                      -sum(TAC) * self._n[-1] * self._ua[-1]*2,
                      -sum(TPC) * self._n[-1] * self._ua[-1]*2,
                      -sum(DC) * self._n[-1] * self._ua[-1]*2])

        return S.squeeze()

    def TSC(self):
        """Third-order transverse spherical aberration"""
        self._precalculations()

        TSC = []
        for k in range(1, self._N-1):
            TSC.append(self._B[k-1] * self._i[k-1]**2 * self._hp)
        return TSC

    def SC(self):
        """Third-order longitudinal spherical aberration"""
        self._precalculations()

        TSC = []
        SC = []
        for k in range(1, self._N-1):
            TSC.append(self._B[k-1] * self._i[k-1]**2 * self._hp)
            SC.append(-TSC[-1] / self._ua[-1])
        return SC

    def CC(self):
        """Third-order sagittal coma"""
        self._precalculations()

        CC = []
        for k in range(1, self._N-1):
            CC.append(self._B[k-1] * self._i[k-1] * self._ip[k-1] * self._hp)
        return CC

    def TCC(self):
        """Third-order tangential coma"""
        return self.CC() * 3

    def TAC(self):
        """Third-order transverse astigmatism"""
        self._precalculations()

        TAC = []
        for k in range(1, self._N-1):
            TAC.append(self._B[k-1] * self._ip[k-1]**2 * self._hp)

        return TAC

    def AC(self):
        """Third-order longitudinal astigmatism"""
        self._precalculations()

        TAC = []
        AC = []
        for k in range(1, self._N-1):
            TAC.append(self._B[k-1] * self._ip[k-1]**2 * self._hp)
            AC.append(-TAC[-1] / self._ua[-1])
        return AC

    def TPC(self):
        """Third-order transverse Petzval sum"""
        self._precalculations()

        TPC = []
        for k in range(1, self._N-1):
            TPC.append((self._n[k] - self._n[k-1]) * self._C[k] * self._hp *
                       self._inv / (2*self._n[k] * self._n[k-1]))
        return TPC

    def PC(self):
        """Third-order longitudinal Petzval sum"""
        self._precalculations()

        TPC = []
        PC = []
        for k in range(1, self._N-1):
            TPC.append((self._n[k] - self._n[k-1]) * self._C[k] * self._hp *
                       self._inv / (2*self._n[k] * self._n[k-1]))
            PC.append(-TPC[-1] / self._ua[-1])
        return PC

    def DC(self):
        """Third-order distortion"""
        self._precalculations()

        DC = []
        for k in range(1, self._N-1):
            DC.append(self._hp * (self._Bp[k-1] * self._i[k-1] * self._ip[k-1] +
                                  0.5*(self._ub[k]**2 - self._ub[k-1]**2)))
        return DC

    def TAchC(self):
        """First-order transverse axial color"""
        self._precalculations()

        TAchC = []
        for k in range(1, self._N-1):
            TAchC.append(-self._ya[k-1] * self._i[k-1] / (self._n[-1] * self._ua[-1]) *
                         (self._dn[k-1] - self._n[k-1] / self._n[k] * self._dn[k]))
        return TAchC

    def LchC(self):
        """First-order longitudinal axial color"""
        self._precalculations()

        TAchC = []
        LchC = []
        for k in range(1, self._N-1):
            TAchC.append(-self._ya[k-1] * self._i[k-1] / (self._n[-1] * self._ua[-1]) *
                         (self._dn[k-1] - self._n[k-1] / self._n[k] * self._dn[k]))
            LchC.append(-TAchC[-1] / self._ua[-1])
        return LchC

    def TchC(self):
        """First-order lateral color"""
        self._precalculations()

        TchC = []
        for k in range(1, self._N-1):
            TchC.append(-self._ya[k-1] * self._ip[k-1] / (self._n[-1] * self._ua[-1]) *
                        (self._dn[k-1] - self._n[k-1] / self._n[k] * self._dn[k]))
        return TchC

    def _precalculations(self):
        self._inv = self.optic.paraxial.invariant()  # Lagrange invariant
        self._n = self.optic.n()  # refractive indices
        self._N = self.optic.surface_group.num_surfaces
        self._C = 1 / self.optic.surface_group.radii
        self._ya, self._ua = self.optic.paraxial.marginal_ray()
        self._yb, self._ub = self.optic.paraxial.chief_ray()
        self._hp = self._inv / (self._n[-1] * self._ua[-1])
        self._dn = self.optic.n(0.4861) - self.optic.n(0.6563)

        self._i = np.zeros(self._N-2)
        self._ip = np.zeros(self._N-2)
        self._B = np.zeros(self._N-2)
        self._Bp = np.zeros(self._N-2)

        for k in range(1, self._N-1):
            self._i[k-1] = self._C[k] * self._ya[k] + self._ua[k-1]
            self._ip[k-1] = self._C[k] * self._yb[k] + self._ub[k-1]
            self._B[k-1] = self._n[k-1] * (self._n[k] - self._n[k-1]) * self._ya[k] * \
                (self._ua[k] + self._i[k-1]) / (2*self._n[k] * self._inv)
            self._Bp[k-1] = self._n[k-1] * (self._n[k] - self._n[k-1]) * self._yb[k] * \
                (self._ub[k] + self._ip[k-1]) / (2*self._n[k] * self._inv)
