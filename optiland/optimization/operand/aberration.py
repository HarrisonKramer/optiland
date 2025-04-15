"""Aberration Operands Module

This module provides a class that calculates various aberration values for an
optical system. It is used in conjunction with the optimization module to
optimize optical systems.

Kramer Harrison, 2024
"""

import optiland.backend as be


class AberrationOperand:
    """A class that provides methods to calculate various aberration values for
        an optic.

    Methods:
        seidels(optic, seidel_number): Returns the seidel aberration value for
            the given seidel number.
        TSC(optic, surface_number): Returns the third-order transverse
            spherical aberration value for the given surface number.
        SC(optic, surface_number): Returns the longitudinal spherical
            aberration value for the given surface number.
        CC(optic, surface_number): Returns the sagittal coma aberration value
            for the given surface number.
        TCC(optic, surface_number): Returns the tangential coma aberration
            value for the given surface number.
        TAC(optic, surface_number): Returns the transverse astigmatism value
            for the given surface number.
        AC(optic, surface_number): Returns the longitudinal astigmatism value
            for the given surface number.
        TPC(optic, surface_number): Returns the transverse Petzval sum
            value for the given surface number.
        PC(optic, surface_number): Returns the longitudinal Petzval sum value
            for the given surface number.
        DC(optic, surface_number): Returns the distortion aberration value for
            the given surface number.
        TAchC(optic, surface_number): Returns the third-order axial chromatic
            aberration value for the given surface number.
        LchC(optic, surface_number): Returns the longitudinal chromatic
            aberration value for the given surface number.
        TchC(optic, surface_number): Returns the transverse chromatic
            aberration value for the given surface number.
        TSC_sum(optic): Returns the sum of third-order transverse
            spherical aberration values for the optic.
        SC_sum(optic): Returns the sum of longitudinal spherical aberration
            values for the optic.
        CC_sum(optic): Returns the sum of sagittal coma
            aberration values for the optic.
        TCC_sum(optic): Returns the sum of tangential coma
            aberration values for the optic.
        TAC_sum(optic): Returns the sum of transverse astigmatism
            aberration values for the optic.
        AC_sum(optic): Returns the sum of longitudinal astigmatism
            aberration values for the optic.
        TPC_sum(optic): Returns the sum of transverse Petzval sum
            aberration values for the optic.
        PC_sum(optic): Returns the sum of longitudinal Petzval sum
            values for the optic.
        DC_sum(optic): Returns the sum of distortion
            aberration values for the optic.
        TAchC_sum(optic): Returns the sum of third-order axial
            chromatic aberration values for the optic.
        LchC_sum(optic): Returns the sum of longitudinal
            chromatic aberration values for the optic.
        TchC_sum(optic): Returns the sum of transverse
            chromatic aberration values for the optic.

    """

    @staticmethod
    def seidels(optic, seidel_number):
        return optic.aberrations.seidels()[seidel_number - 1]

    @staticmethod
    def TSC(optic, surface_number):
        return optic.aberrations.TSC()[surface_number]

    @staticmethod
    def SC(optic, surface_number):
        return optic.aberrations.SC()[surface_number]

    @staticmethod
    def CC(optic, surface_number):
        return optic.aberrations.CC()[surface_number]

    @staticmethod
    def TCC(optic, surface_number):
        return optic.aberrations.TCC()[surface_number]

    @staticmethod
    def TAC(optic, surface_number):
        return optic.aberrations.TAC()[surface_number]

    @staticmethod
    def AC(optic, surface_number):
        return optic.aberrations.AC()[surface_number]

    @staticmethod
    def TPC(optic, surface_number):
        return optic.aberrations.TPC()[surface_number]

    @staticmethod
    def PC(optic, surface_number):
        return optic.aberrations.PC()[surface_number]

    @staticmethod
    def DC(optic, surface_number):
        return optic.aberrations.DC()[surface_number]

    @staticmethod
    def TAchC(optic, surface_number):
        return optic.aberrations.TAchC()[surface_number]

    @staticmethod
    def LchC(optic, surface_number):
        return optic.aberrations.LchC()[surface_number]

    @staticmethod
    def TchC(optic, surface_number):
        return optic.aberrations.TchC()[surface_number]

    @staticmethod
    def TSC_sum(optic):
        return be.sum(optic.aberrations.TSC())

    @staticmethod
    def SC_sum(optic):
        return be.sum(optic.aberrations.SC())

    @staticmethod
    def CC_sum(optic):
        return be.sum(optic.aberrations.CC())

    @staticmethod
    def TCC_sum(optic):
        return be.sum(optic.aberrations.TCC())

    @staticmethod
    def TAC_sum(optic):
        return be.sum(optic.aberrations.TAC())

    @staticmethod
    def AC_sum(optic):
        return be.sum(optic.aberrations.AC())

    @staticmethod
    def TPC_sum(optic):
        return be.sum(optic.aberrations.TPC())

    @staticmethod
    def PC_sum(optic):
        return be.sum(optic.aberrations.PC())

    @staticmethod
    def DC_sum(optic):
        return be.sum(optic.aberrations.DC())

    @staticmethod
    def TAchC_sum(optic):
        return be.sum(optic.aberrations.TAchC())

    @staticmethod
    def LchC_sum(optic):
        return be.sum(optic.aberrations.LchC())

    @staticmethod
    def TchC_sum(optic):
        return be.sum(optic.aberrations.TchC())
