import numpy as np
from optiland import wavefront
from optiland.distribution import GaussianQuadrature


class ParaxialOperand:

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
    def TSC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TSC()[surface_number])

    @staticmethod
    def SC_sum(optic, surface_number):
        return np.sum(optic.aberrations.SC()[surface_number])

    @staticmethod
    def CC_sum(optic, surface_number):
        return np.sum(optic.aberrations.CC()[surface_number])

    @staticmethod
    def TCC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TCC()[surface_number])

    @staticmethod
    def TAC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TAC()[surface_number])

    @staticmethod
    def AC_sum(optic, surface_number):
        return np.sum(optic.aberrations.AC()[surface_number])

    @staticmethod
    def TPC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TPC()[surface_number])

    @staticmethod
    def PC_sum(optic, surface_number):
        return np.sum(optic.aberrations.PC()[surface_number])

    @staticmethod
    def DC_sum(optic, surface_number):
        return np.sum(optic.aberrations.DC()[surface_number])

    @staticmethod
    def TAchC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TAchC()[surface_number])

    @staticmethod
    def LchC_sum(optic, surface_number):
        return np.sum(optic.aberrations.LchC()[surface_number])

    @staticmethod
    def TchC_sum(optic, surface_number):
        return np.sum(optic.aberrations.TchC()[surface_number])


class RayOperand:

    @staticmethod
    def x_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.x[surface_number, 0]

    @staticmethod
    def y_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.y[surface_number, 0]

    @staticmethod
    def z_intercept(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.z[surface_number, 0]

    @staticmethod
    def L(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.L[surface_number, 0]

    @staticmethod
    def M(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.M[surface_number, 0]

    @staticmethod
    def N(optic, surface_number, Hx, Hy, Px, Py, wavelength):
        optic.trace_generic(Hx, Hy, Px, Py, wavelength)
        return optic.surface_group.N[surface_number, 0]

    @staticmethod
    def rms_spot_size(optic, surface_number, Hx, Hy, num_rays, wavelength,
                      distribution='hexapolar'):
        optic.trace(Hx, Hy, wavelength, num_rays, distribution)
        x = optic.surface_group.x[surface_number, :].flatten()
        y = optic.surface_group.x[surface_number, :].flatten()
        r2 = x**2 + y**2
        return np.sqrt(np.mean(r2))

    @staticmethod
    def OPD_difference(optic, Hx, Hy, num_rays, wavelength,
                       distribution='gaussian_quad'):
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

    def __init__(self, operand_type, target, weight, input_data={},
                 metric_dict=METRIC_DICT):
        self.operand_type = operand_type
        self.target = target
        self.weight = weight
        self.input_data = input_data
        self.metric_dict = metric_dict

    @property
    def value(self):
        '''Get current value of the operand'''
        metric_function = self.metric_dict.get(self.operand_type)
        if metric_function:
            return metric_function(**self.input_data)
        else:
            raise ValueError(f'Unknown operand type: {self.operand_type}')

    def delta(self):
        '''delta between target and current value'''
        return (self.value - self.target)

    def fun(self):
        '''return objective function value'''
        return self.weight * self.delta()

    def info(self, number=None):
        if number is not None:
            print(f'\tOperand {number}')
        print(f'\t   Type: {self.operand_type}')
        print(f'\t   Weight: {self.weight}')
        print(f'\t   Target: {self.target}')
        print(f'\t   Value: {self.value}')
        print(f'\t   Delta: {self.delta()}')
