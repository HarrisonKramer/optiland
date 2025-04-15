"""
Aberrations Module

This module computes first- and third-order aberrations of optical systems.
The aberration calculations are based on the algorithms outlined in
Modern Optical Engineering by Warren Smith (Chapter 6.3).

Kramer Harrison, 2023
"""

import optiland.backend as be


class Aberrations:
    """Class for computation of optical aberrations.

    This class provides methods to compute first- and third-order aberrations
    of a general optical system, using an instance of an optic.Optic object.

    Args:
        optic (Optic): An instance of the optical system to be analyzed.
    """

    def __init__(self, optic):
        self.optic = optic

    def third_order(self):
        """
        Compute all third-order aberrations and first-order color terms.

        Returns:
            tuple: A tuple containing the following numpy arrays:
                - TSC: Third-order transverse spherical aberration.
                - SC: Third-order longitudinal spherical aberration.
                - CC: Third-order sagittal coma.
                - TCC: Third-order tangential coma (3×CC).
                - TAC: Third-order transverse astigmatism.
                - AC: Third-order longitudinal astigmatism.
                - TPC: Third-order transverse Petzval sum.
                - PC: Third-order longitudinal Petzval sum.
                - DC: Third-order distortion.
                - TAchC: First-order transverse axial color.
                - LchC: First-order longitudinal axial color.
                - TchC: First-order lateral color.
                - S: Seidel aberration coefficients.
        """
        self._precalculations()

        # Compute aberration terms over surfaces 1 to N-2
        TSC = self._compute_over_surfaces(self._TSC_term)
        CC = self._compute_over_surfaces(self._CC_term)
        TAC = self._compute_over_surfaces(self._TAC_term)
        TPC = self._compute_over_surfaces(self._TPC_term)
        DC = self._compute_over_surfaces(self._DC_term)
        TAchC = self._compute_over_surfaces(self._TAchC_term)
        TchC = self._compute_over_surfaces(self._TchC_term)

        # Compute derived terms using the computed aberration values
        SC = -TSC / self._ua[-1]
        AC = -TAC / self._ua[-1]
        PC = -TPC / self._ua[-1]
        LchC = -TAchC / self._ua[-1]

        S = self._sum_seidels(TSC, CC, TAC, TPC, DC)
        TCC = CC * 3

        return (
            TSC.flatten(),
            SC.flatten(),
            CC.flatten(),
            TCC.flatten(),
            TAC.flatten(),
            AC.flatten(),
            TPC.flatten(),
            PC.flatten(),
            DC.flatten(),
            TAchC.flatten(),
            LchC.flatten(),
            TchC.flatten(),
            S,
        )

    def seidels(self):
        """
        Compute the Seidel aberration coefficients.

        Returns:
            numpy.ndarray: Array of Seidel aberration coefficients.
        """
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        CC = self._compute_over_surfaces(self._CC_term)
        TAC = self._compute_over_surfaces(self._TAC_term)
        TPC = self._compute_over_surfaces(self._TPC_term)
        DC = self._compute_over_surfaces(self._DC_term)
        S = self._sum_seidels(TSC, CC, TAC, TPC, DC)
        return S.squeeze()

    def TSC(self):
        """
        Compute third-order transverse spherical aberration.

        Returns:
            numpy.ndarray: Third-order transverse spherical aberration.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TSC_term).flatten()

    def SC(self):
        """
        Compute third-order longitudinal spherical aberration.

        Returns:
            numpy.ndarray: Third-order longitudinal spherical aberration.
        """
        self._precalculations()
        TSC = self._compute_over_surfaces(self._TSC_term)
        SC = -TSC / self._ua[-1]
        return SC.flatten()

    def CC(self):
        """
        Compute third-order sagittal coma.

        Returns:
            numpy.ndarray: Third-order sagittal coma.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._CC_term).flatten()

    def TCC(self):
        """
        Compute third-order tangential coma.

        Returns:
            numpy.ndarray: Third-order tangential coma.
        """
        return (self.CC() * 3).flatten()

    def TAC(self):
        """
        Compute third-order transverse astigmatism.

        Returns:
            numpy.ndarray: Third-order transverse astigmatism.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TAC_term).flatten()

    def AC(self):
        """
        Compute third-order longitudinal astigmatism.

        Returns:
            numpy.ndarray: Third-order longitudinal astigmatism.
        """
        self._precalculations()
        TAC = self._compute_over_surfaces(self._TAC_term)
        AC = -TAC / self._ua[-1]
        return AC.flatten()

    def TPC(self):
        """
        Compute third-order transverse Petzval sum.

        Returns:
            numpy.ndarray: Third-order transverse Petzval sum.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TPC_term).flatten()

    def PC(self):
        """
        Compute third-order longitudinal Petzval sum.

        Returns:
            numpy.ndarray: Third-order longitudinal Petzval sum.
        """
        self._precalculations()
        TPC = self._compute_over_surfaces(self._TPC_term)
        PC = -TPC / self._ua[-1]
        return PC.flatten()

    def DC(self):
        """
        Compute third-order distortion.

        Returns:
            numpy.ndarray: Third-order distortion.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._DC_term).flatten()

    def TAchC(self):
        """
        Compute first-order transverse axial color.

        Returns:
            numpy.ndarray: First-order transverse axial color.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TAchC_term).flatten()

    def LchC(self):
        """
        Compute first-order longitudinal axial color.

        Returns:
            numpy.ndarray: First-order longitudinal axial color.
        """
        self._precalculations()
        TAchC = self._compute_over_surfaces(self._TAchC_term)
        LchC = -TAchC / self._ua[-1]
        return LchC.flatten()

    def TchC(self):
        """
        Compute first-order lateral color.

        Returns:
            numpy.ndarray: First-order lateral color.
        """
        self._precalculations()
        return self._compute_over_surfaces(self._TchC_term).flatten()

    def _compute_over_surfaces(self, term_func):
        """
        Compute a given aberration term over all relevant surfaces.

        Args:
            term_func (callable): Function that computes a term for a given
                surface index.

        Returns:
            numpy.ndarray: Array of computed term values over surfaces 1 to N-2.
        """
        terms = [term_func(k) for k in range(1, self._N - 1)]
        return be.array(terms)

    def _precalculations(self):
        """
        Perform all necessary precalculations for aberration computations.

        This method computes and stores common parameters required by the
        various aberration term methods.
        """
        self._inv = self.optic.paraxial.invariant()  # Lagrange invariant
        self._n = self.optic.n()  # Refractive indices for all surfaces
        self._N = self.optic.surface_group.num_surfaces
        self._C = 1 / self.optic.surface_group.radii
        self._ya, self._ua = self.optic.paraxial.marginal_ray()
        self._yb, self._ub = self.optic.paraxial.chief_ray()
        self._hp = self._inv / (self._n[-1] * self._ua[-1])
        self._dn = self.optic.n(0.4861) - self.optic.n(0.6563)

        # Initialize arrays for intermediate variables over surfaces 1 to N-2.
        num = self._N - 2
        self._i = be.zeros(num)
        self._ip = be.zeros(num)
        self._B = be.zeros(num)
        self._Bp = be.zeros(num)

        for k in range(1, self._N - 1):
            idx = k - 1
            # Calculate intermediate angles
            self._i[idx] = (self._C[k] * self._ya[k] + self._ua[k - 1])[0]
            self._ip[idx] = (self._C[k] * self._yb[k] + self._ub[k - 1])[0]

            denom = 2 * self._n[k] * self._inv
            if denom == 0:
                self._B[idx] = 0
                self._Bp[idx] = 0
            else:
                self._B[idx] = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._ya[k]
                    * (self._ua[k] + self._i[idx])
                    / denom
                )[0]
                self._Bp[idx] = (
                    self._n[k - 1]
                    * (self._n[k] - self._n[k - 1])
                    * self._yb[k]
                    * (self._ub[k] + self._ip[idx])
                    / denom
                )[0]

    def _TSC_term(self, k):
        """
        Compute third-order transverse spherical aberration term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed transverse spherical aberration term.
        """
        return self._B[k - 1] * self._i[k - 1] ** 2 * self._hp

    def _CC_term(self, k):
        """
        Compute third-order sagittal coma term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed sagittal coma term.
        """
        return self._B[k - 1] * self._i[k - 1] * self._ip[k - 1] * self._hp

    def _TAC_term(self, k):
        """
        Compute third-order transverse astigmatism term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed transverse astigmatism term.
        """
        return self._B[k - 1] * self._ip[k - 1] ** 2 * self._hp

    def _TPC_term(self, k):
        """
        Compute third-order transverse Petzval sum term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed transverse Petzval sum term.
        """
        return (
            (self._n[k] - self._n[k - 1])
            * self._C[k]
            * self._hp
            * self._inv
            / (2 * self._n[k] * self._n[k - 1])
        )

    def _DC_term(self, k):
        """
        Compute third-order distortion term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed distortion term.
        """
        return self._hp * (
            self._Bp[k - 1] * self._i[k - 1] * self._ip[k - 1]
            + 0.5 * (self._ub[k] ** 2 - self._ub[k - 1] ** 2)
        )

    def _TAchC_term(self, k):
        """
        Compute first-order transverse axial color term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed transverse axial color term.
        """
        return (
            -self._ya[k - 1]
            * self._i[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _TchC_term(self, k):
        """
        Compute first-order lateral color term for surface k.

        Args:
            k (int): Surface index.

        Returns:
            float: Computed lateral color term.
        """
        return (
            -self._ya[k - 1]
            * self._ip[k - 1]
            / (self._n[-1] * self._ua[-1])
            * (self._dn[k - 1] - self._n[k - 1] / self._n[k] * self._dn[k])
        )

    def _sum_seidels(self, TSC, CC, TAC, TPC, DC):
        """
        Sum the Seidel aberration coefficients from the individual terms.

        Args:
            TSC (numpy.ndarray): Transverse spherical aberration terms.
            CC (numpy.ndarray): Sagittal coma terms.
            TAC (numpy.ndarray): Transverse astigmatism terms.
            TPC (numpy.ndarray): Transverse Petzval sum terms.
            DC (numpy.ndarray): Distortion terms.

        Returns:
            numpy.ndarray: Array of Seidel aberration coefficients.
        """
        factor = self._n[-1] * self._ua[-1] * 2
        return be.array(
            [
                -be.sum(TSC) * factor,
                -be.sum(CC) * factor,
                -be.sum(TAC) * factor,
                -be.sum(TPC) * factor,
                -be.sum(DC) * factor,
            ]
        )
