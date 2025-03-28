"""Aberrations Module

This module computes first and third order aberrations of optical systems.

Kramer Harrison, 2023
"""

import numpy as np


class Aberrations:
    """Aberrations class for computation of optical aberrations

    This class provides an interface to compute first and third order
    aberrations of general optical systems, as defined in a optic.Optic
    instance.

    Most calculations in this class are based on the algorithms outlined in
    Modern Optical Engineering, Warren Smith, Chapter 6.3.

    Attributes:
        optic (optic.Optic): instance of the optic object to be assessed

    """

    def __init__(self, optic):
        """Create an instance of the Aberration class

        Args:
            optic (optic.Optic): instance of the optic object to be assessed

        Returns:
            None

        """
        self.optic = optic

    def third_order(self):
        """Compute all third order aberrations

        Returns:
            TSC (List[float]): Third-order transverse spherical aberration
            SC (List[float]): Third-order longitudinal spherical aberration
            CC (List[float]): Third-order sagittal coma
            TCC (List[float]): Third-order tangential coma
            TAC (List[float]): Third-order transverse astigmatism
            AC (List[float]): Third-order longitudinal astigmatism
            TPC (List[float]): Third-order transverse Petzval sum
            PC (List[float]): Third-order longitudinal Petzval sum
            DC (List[float]): Third-order distortion
            TAchC (List[float]): First-order transverse axial color
            LchC (List[float]): First-order longitudinal axial color
            TchC (List[float]): First-order lateral color
            S (List[float]): Seidel aberration coefficients

        """
        self._precalculations()

        SC = []
        AC = []
        PC = []
        TAchC = []
        LchC = []
        TchC = []

        TSC, CC, TAC, TPC, DC = self._compute_seidel_terms()

        for k in range(1, self._N - 1):
            TAchC.append(self._TAchC_term(k))
            TchC.append(self._TchC_term(k))

            SC.append(-TSC[k - 1] / self._ua[-1])
            AC.append(-TAC[k - 1] / self._ua[-1])
            PC.append(-TPC[k - 1] / self._ua[-1])
            LchC.append(-TAchC[k - 1] / self._ua[-1])

        S = self._sum_seidels([TSC, CC, TAC, TPC, DC])

        TSC = np.array(TSC).flatten()
        CC = np.array(CC).flatten()
        TAC = np.array(TAC).flatten()
        TPC = np.array(TPC).flatten()
        DC = np.array(DC).flatten()
        TAchC = np.array(TAchC).flatten()
        TchC = np.array(TchC).flatten()
        AC = np.array(AC).flatten()
        PC = np.array(PC).flatten()
        LchC = np.array(LchC).flatten()

        return TSC, SC, CC, CC * 3, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S

    def seidels(self):
        """Compute the seidel aberration coefficients

        Returns:
            S (List[float]): Seidel aberration coefficients

        """
        self._precalculations()
        TSC, CC, TAC, TPC, DC = self._compute_seidel_terms()
        S = self._sum_seidels([TSC, CC, TAC, TPC, DC])
        return S.squeeze()

    def TSC(self):
        """Compute third-order transverse spherical aberration

        Returns:
            TSC (List[float]): Third-order transverse spherical aberration

        """
        self._precalculations()

        TSC = []
        for k in range(1, self._N - 1):
            TSC.append(self._TSC_term(k))
        return np.array(TSC).flatten()

    def SC(self):
        """Compute third-order longitudinal spherical aberration

        Returns:
            SC (List[float]): Third-order longitudinal spherical aberration

        """
        self._precalculations()

        TSC = []
        SC = []
        for k in range(1, self._N - 1):
            TSC.append(self._TSC_term(k))
            SC.append(-TSC[-1] / self._ua[-1])
        return np.array(SC).flatten()

    def CC(self):
        """Compute third-order sagittal coma

        Returns:
            CC (List[float]): Third-order sagittal coma

        """
        self._precalculations()

        CC = []
        for k in range(1, self._N - 1):
            CC.append(self._CC_term(k))
        return np.array(CC).flatten()

    def TCC(self):
        """Compute third-order tangential coma

        Returns:
            TCC (List[float]): Third-order tangential coma

        """
        return self.CC() * 3

    def TAC(self):
        """Compute third-order transverse astigmatism

        Returns:
            TAC (List[float]): Third-order transverse astigmatism

        """
        self._precalculations()

        TAC = []
        for k in range(1, self._N - 1):
            TAC.append(self._TAC_term(k))

        return np.array(TAC).flatten()

    def AC(self):
        """Compute third-order longitudinal astigmatism

        Returns:
            AC (List[float]): Third-order longitudinal astigmatism

        """
        self._precalculations()

        TAC = []
        AC = []
        for k in range(1, self._N - 1):
            TAC.append(self._TAC_term(k))
            AC.append(-TAC[-1] / self._ua[-1])
        return np.array(AC).flatten()

    def TPC(self):
        """Compute third-order transverse Petzval sum

        Returns:
            TPC (List[float]): Third-order transverse Petzval sum

        """
        self._precalculations()

        TPC = []
        for k in range(1, self._N - 1):
            TPC.append(self._TPC_term(k))
        return np.array(TPC).flatten()

    def PC(self):
        """Compute third-order longitudinal Petzval sum

        Returns:
            PC (List[float]): Third-order longitudinal Petzval sum

        """
        self._precalculations()

        TPC = []
        PC = []
        for k in range(1, self._N - 1):
            TPC.append(self._TPC_term(k))
            PC.append(-TPC[-1] / self._ua[-1])
        return np.array(PC).flatten()

    def DC(self):
        """Compute third-order distortion

        Returns:
            DC (List[float]): Third-order distortion

        """
        self._precalculations()

        DC = []
        for k in range(1, self._N - 1):
            DC.append(self._DC_term(k))
        return np.array(DC).flatten()

    def TAchC(self):
        """Compute first-order transverse axial color

        Returns:
            TAchC (List[float]): First-order transverse axial color

        """
        self._precalculations()

        TAchC = []
        for k in range(1, self._N - 1):
            TAchC.append(self._TAchC_term(k))
        return np.array(TAchC).flatten()

    def LchC(self):
        """Compute first-order longitudinal axial color

        Returns:
            LchC (List[float]): First-order longitudinal axial color

        """
        self._precalculations()

        TAchC = []
        LchC = []
        for k in range(1, self._N - 1):
            TAchC.append(self._TAchC_term(k))
            LchC.append(-TAchC[-1] / self._ua[-1])
        return np.array(LchC).flatten()

    def TchC(self):
        """Compute first-order lateral color

        Returns:
            TchC (List[float]): First-order lateral color

        """
        self._precalculations()

        TchC = []
        for k in range(1, self._N - 1):
            TchC.append(self._TchC_term(k))
        return np.array(TchC).flatten()

    def _TAchC_term(self, k):
        """Compute first-order transverse axial color term"""
        return (
            -self._ya[k - 1]
            * self._i[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _TchC_term(self, k):
        """Compute first-order lateral color term"""
        return (
            -self._ya[k - 1]
            * self._ip[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _TSC_term(self, k):
        """Compute third-order transverse spherical aberration term"""
        return self._B[k - 1] * self._i[k - 1] ** 2 * self._hp

    def _CC_term(self, k):
        """Compute third-order sagittal coma term"""
        return self._B[k - 1] * self._i[k - 1] * self._ip[k - 1] * self._hp

    def _TAC_term(self, k):
        """Compute third-order transverse astigmatism term"""
        return self._B[k - 1] * self._ip[k - 1] ** 2 * self._hp

    def _TPC_term(self, k):
        """Compute third-order transverse Petzval sum term"""
        return (
            (self._n[k] - self._n[k - 1])
            * self._C[k]
            * self._hp
            * self._inv
            / (2 * self._n[k] * self._n[k - 1])
        )

    def _DC_term(self, k):
        """Compute third-order distortion term"""
        return self._hp * (
            self._Bp[k - 1] * self._i[k - 1] * self._ip[k - 1]
            + 0.5 * (self._ub[k] ** 2 - self._ub[k - 1] ** 2)
        )

    def _precalculations(self):
        """Perform precalculations needed for most aberration calculations"""
        self._inv = self.optic.paraxial.invariant()  # Lagrange invariant
        self._n = self.optic.n()  # refractive indices
        self._N = self.optic.surface_group.num_surfaces
        self._C = 1 / self.optic.surface_group.radii
        self._ya, self._ua = self.optic.paraxial.marginal_ray()
        self._yb, self._ub = self.optic.paraxial.chief_ray()
        self._hp = self._inv / (self._n[-1] * self._ua[-1])
        self._dn = self.optic.n(0.4861) - self.optic.n(0.6563)

        self._i = np.zeros(self._N - 2)
        self._ip = np.zeros(self._N - 2)
        self._B = np.zeros(self._N - 2)
        self._Bp = np.zeros(self._N - 2)

        for k in range(1, self._N - 1):
            self._i[k - 1] = (self._C[k] * self._ya[k] + self._ua[k - 1])[0]
            self._ip[k - 1] = (self._C[k] * self._yb[k] + self._ub[k - 1])[0]

            denom = 2 * self._n[k] * self._inv
            if denom == 0:
                self._B[k - 1] = 0
                self._Bp[k - 1] = 0
            else:
                self._B[k - 1] = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._ya[k]
                    * (self._ua[k] + self._i[k - 1])
                    / denom
                )[0]
                self._Bp[k - 1] = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._yb[k]
                    * (self._ub[k] + self._ip[k - 1])
                    / denom
                )[0]

    def _compute_seidel_terms(self):
        """Compute the Seidel aberration terms"""
        TSC = []
        CC = []
        TAC = []
        TPC = []
        DC = []

        for k in range(1, self._N - 1):
            TSC.append(self._TSC_term(k))
            CC.append(self._CC_term(k))
            TAC.append(self._TAC_term(k))
            TPC.append(self._TPC_term(k))
            DC.append(self._DC_term(k))

        return TSC, CC, TAC, TPC, DC

    def _sum_seidels(self, seidels):
        """Sum the Seidel aberration coefficients"""
        TSC, CC, TAC, TPC, DC = seidels
        S = np.array(
            [
                -sum(TSC) * self._n[-1] * self._ua[-1] * 2,
                -sum(CC) * self._n[-1] * self._ua[-1] * 2,
                -sum(TAC) * self._n[-1] * self._ua[-1] * 2,
                -sum(TPC) * self._n[-1] * self._ua[-1] * 2,
                -sum(DC) * self._n[-1] * self._ua[-1] * 2,
            ],
        )
        return S
