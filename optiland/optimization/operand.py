"""Optiland Operand Module

This module gives various operands to be used during lens optimization. These
include paraxial, real ray, aberrations, wavefront, spot size, and many other
operand types.

In general, to use one of these operands in optimization, you
can simply do the following:
    1. Identify the key in the METRIC_DICT variable, or add your new operand.
    2. Identify the input data dictionary that is required for the calculation
       of this operand.
    3. Add the operand to your optimization.OptimizationProblem instance using
       the add_operand method. Include the operand type, target, weight, and
       the input data (in dict format). See examples.

Kramer Harrison, 2024
"""
import numpy as np
from optiland import wavefront
from optiland.distribution import GaussianQuadrature


class ParaxialOperand:
    """
    A class representing paraxial operands.

    This class provides static methods to calculate various paraxial
        properties of an optic.

    Attributes:
        None

    Methods:
        f1(optic): Returns the first focal length of the optic.
        f2(optic): Returns the second focal length of the optic.
        F1(optic): Returns the first principal plane distance of the optic.
        F2(optic): Returns the second principal plane distance of the optic.
        P1(optic): Returns the first principal point distance of the optic.
        P2(optic): Returns the second principal point distance of the optic.
        N1(optic): Returns the first nodal point distance of the optic.
        N2(optic): Returns the second nodal point distance of the optic.
        EPL(optic): Returns the entrance pupil distance of the optic.
        EPD(optic): Returns the entrance pupil diameter of the optic.
        XPL(optic): Returns the exit pupil distance of the optic.
        XPD(optic): Returns the exit pupil diameter of the optic.
        magnification(optic): Returns the magnification of the optic.
    """

    @staticmethod
    def f1(optic):
        return optic.paraxial.f1()

    @staticmethod
    def f2(optic):
        return optic.paraxial.f2()

    @staticmethod
    def F1(optic):
        return optic.paraxial.F1()

    @staticmethod
    def F2(optic):
        return optic.paraxial.F2()

    @staticmethod
    def P1(optic):
        return optic.paraxial.P1()

    @staticmethod
    def P2(optic):
        return optic.paraxial.P2()

    @staticmethod
    def N1(optic):
        return optic.paraxial.N1()

    @staticmethod
    def N2(optic):
        return optic.paraxial.N2()

    @staticmethod
    def EPL(optic):
        return optic.paraxial.EPL()

    @staticmethod
    def EPD(optic):
        return optic.paraxial.EPD()

    @staticmethod
    def XPL(optic):
        return optic.paraxial.XPL()

    @staticmethod
    def XPD(optic):
        return optic.paraxial.XPD()

    @staticmethod
    def magnification(optic):
        return optic.paraxial.magnification()


class AberrationOperand:
    """
    A class that provides methods to calculate various aberration values for
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
        return optic.aberrations.seidels()[seidel_number-1]

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
        return np.sum(optic.aberrations.TSC())

    @staticmethod
    def SC_sum(optic):
        return np.sum(optic.aberrations.SC())

    @staticmethod
    def CC_sum(optic):
        return np.sum(optic.aberrations.CC())

    @staticmethod
    def TCC_sum(optic):
        return np.sum(optic.aberrations.TCC())

    @staticmethod
    def TAC_sum(optic):
        return np.sum(optic.aberrations.TAC())

    @staticmethod
    def AC_sum(optic):
        return np.sum(optic.aberrations.AC())

    @staticmethod
    def TPC_sum(optic):
        return np.sum(optic.aberrations.TPC())

    @staticmethod
    def PC_sum(optic):
        return np.sum(optic.aberrations.PC())

    @staticmethod
    def DC_sum(optic):
        return np.sum(optic.aberrations.DC())

    @staticmethod
    def TAchC_sum(optic):
        return np.sum(optic.aberrations.TAchC())

    @staticmethod
    def LchC_sum(optic):
        return np.sum(optic.aberrations.LchC())

    @staticmethod
    def TchC_sum(optic):
        return np.sum(optic.aberrations.TchC())


class RayOperand:
    """
    A class that provides static methods for performing ray tracing
        calculations on an optic.

    Methods:
        x_intercept: Calculates the x-coordinate of the intercept point on a
            specific surface.
        y_intercept: Calculates the y-coordinate of the intercept point on a
            specific surface.
        z_intercept: Calculates the z-coordinate of the intercept point on a
            specific surface.
        L: Calculates the direction cosine L of the ray on a specific surface.
        M: Calculates the direction cosine M of the ray on a specific surface.
        N: Calculates the direction cosine N of the ray on a specific surface.
        rms_spot_size: Calculates the root mean square (RMS) spot size on a
            specific surface.
        OPD_difference: Calculates the optical path difference (OPD)
            difference for a given ray distribution.
    """

    @staticmethod
    def x_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the x-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The x-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.x[surface_number, 0]

    @staticmethod
    def y_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the y-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The y-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.y[surface_number, 0]

    @staticmethod
    def z_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the z-coordinate of the intercept point on a specific
            surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The z-coordinate of the intercept point.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.z[surface_number, 0]

    @staticmethod
    def L(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine L of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine L of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.L[surface_number, 0]

    @staticmethod
    def M(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine M of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine M of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.M[surface_number, 0]

    @staticmethod
    def N(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        """
        Calculates the direction cosine N of the ray on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            Px: The x-coordinate of the point on the surface.
            Py: The y-coordinate of the point on the surface.
            wavelength: The wavelength of the ray.

        Returns:
            The direction cosine N of the ray.
        """
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.N[surface_number, 0]

    @staticmethod
    def rms_spot_size(optic, surface_number, Hx, Hy, num_rays, wavelength,
                      distribution='hexapolar'):
        """
        Calculates the root mean square (RMS) spot size on a specific surface.

        Parameters:
            optic: The optic object.
            surface_number: The number of the surface.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays. Default is 'hexapolar'.

        Returns:
            The RMS spot size on the specified surface.
        """
        if wavelength == 'all':
            x = []
            y = []
            for wave in optic.wavelengths.get_wavelengths():
                optic.trace(Hx, Hy, wave, num_rays, distribution)
                x.append(optic.surface_group.x[surface_number, :].flatten())
                y.append(optic.surface_group.y[surface_number, :].flatten())
            wave_idx = optic.wavelengths.primary_index
            mean_x = np.mean(x[wave_idx])
            mean_y = np.mean(y[wave_idx])
            r2 = [(x[i] - mean_x)**2 + (y[i] - mean_y)**2
                  for i in range(len(x))]
            return np.sqrt(np.mean(np.concatenate(r2)))
        else:
            optic.trace(Hx, Hy, wavelength, num_rays, distribution)
            x = optic.surface_group.x[surface_number, :].flatten()
            y = optic.surface_group.y[surface_number, :].flatten()
            r2 = (x - np.mean(x))**2 + (y - np.mean(y))**2
            return np.sqrt(np.mean(r2))

    @staticmethod
    def OPD_difference(optic, Hx, Hy, num_rays, wavelength,
                       distribution='gaussian_quad'):
        """
        Calculates the mean optical path difference (OPD) difference for a
            given ray distribution.

        Parameters:
            optic: The optic object.
            Hx: The x-coordinate of the incoming ray direction.
            Hy: The y-coordinate of the incoming ray direction.
            num_rays: The number of rays to trace.
            wavelength: The wavelength of the rays.
            distribution: The distribution of the rays.
                Default is 'gaussian_quad'.

        Returns:
            The OPD difference for the given ray distribution.
        """
        weights = 1.0

        if distribution == 'gaussian_quad':
            if Hx == Hy == 0:
                distribution = GaussianQuadrature(is_symmetric=True)
                weights = distribution.get_weights(num_rays)
            else:
                distribution = GaussianQuadrature(is_symmetric=False)
                weights = np.repeat(distribution.get_weights(num_rays), 3)

            distribution.generate_points(num_rings=num_rays)

        wf = wavefront.Wavefront(optic, [(Hx, Hy)], [wavelength], num_rays,
                                 distribution)
        delta = (wf.data[0][0][0] - np.mean(wf.data[0][0][0])) * weights
        return np.mean(np.abs(delta))


METRIC_DICT = {
    'f1': ParaxialOperand.f1,
    'f2': ParaxialOperand.f2,
    'F1': ParaxialOperand.F1,
    'F2': ParaxialOperand.F2,
    'P1': ParaxialOperand.P1,
    'P2': ParaxialOperand.P2,
    'N1': ParaxialOperand.N1,
    'N2': ParaxialOperand.N2,
    'EPD': ParaxialOperand.EPD,
    'EPL': ParaxialOperand.EPL,
    'XPD': ParaxialOperand.XPD,
    'XPL': ParaxialOperand.XPL,
    'magnification': ParaxialOperand.magnification,
    'seidel': AberrationOperand.seidels,
    'TSC': AberrationOperand.TSC,
    'SC': AberrationOperand.SC,
    'CC': AberrationOperand.CC,
    'TCC': AberrationOperand.TCC,
    'TAC': AberrationOperand.TAC,
    'AC': AberrationOperand.AC,
    'TPC': AberrationOperand.TPC,
    'PC': AberrationOperand.PC,
    'DC': AberrationOperand.DC,
    'TAchC': AberrationOperand.TAchC,
    'LchC': AberrationOperand.LchC,
    'TchC': AberrationOperand.TchC,
    'TSC_sum': AberrationOperand.TSC,
    'SC_sum': AberrationOperand.SC,
    'CC_sum': AberrationOperand.CC,
    'TCC_sum': AberrationOperand.TCC,
    'TAC_sum': AberrationOperand.TAC,
    'AC_sum': AberrationOperand.AC,
    'TPC_sum': AberrationOperand.TPC,
    'PC_sum': AberrationOperand.PC,
    'DC_sum': AberrationOperand.DC,
    'TAchC_sum': AberrationOperand.TAchC,
    'LchC_sum': AberrationOperand.LchC,
    'TchC_sum': AberrationOperand.TchC,
    'real_x_intercept': RayOperand.x_intercept,
    'real_y_intercept': RayOperand.y_intercept,
    'real_z_intercept': RayOperand.z_intercept,
    'real_L': RayOperand.L,
    'real_M': RayOperand.M,
    'real_N': RayOperand.N,
    'rms_spot_size': RayOperand.rms_spot_size,
    'OPD_difference': RayOperand.OPD_difference
}


class Operand(object):
    """
    Represents an operand used in optimization calculations.

    Attributes:
        type (str): The type of the operand.
        target (float): The target value for the operand.
        weight (float): The weight of the operand.
        input_data (dict): Additional input data for the operand's metric
            function.
        metric_dict (dict): A dictionary mapping operand types to metric
            functions.

    Methods:
        value(): Get the current value of the operand.
        delta(): Calculate the difference between the target and current value.
        fun(): Calculate the objective function value.
    """

    def __init__(self, operand_type, target, weight, input_data={},
                 metric_dict=METRIC_DICT):
        self.type = operand_type
        self.target = target
        self.weight = weight
        self.input_data = input_data
        self.metric_dict = metric_dict

    @property
    def value(self):
        """Get current value of the operand"""
        metric_function = self.metric_dict.get(self.type)
        if metric_function:
            return metric_function(**self.input_data)
        else:
            raise ValueError(f'Unknown operand type: {self.type}')

    def delta(self):
        """Calculate the difference between the target and current value"""
        return (self.value - self.target)

    def fun(self):
        """Calculate the objective function value"""
        return self.weight * self.delta()
