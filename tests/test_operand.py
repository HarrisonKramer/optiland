import pytest
import numpy as np
from optiland import operand
from optiland.samples.telescopes import HubbleTelescope


@pytest.fixture
def hubble():
    return HubbleTelescope()


class TestParaxialOperand:
    def test_f1(self, hubble):
        assert np.isclose(operand.ParaxialOperand.f1(hubble),
                          -57600.08099840297)

    def test_f2(self, hubble):
        assert np.isclose(operand.ParaxialOperand.f2(hubble),
                          57600.080998403595)

    def test_F1(self, hubble):
        assert np.isclose(operand.ParaxialOperand.F1(hubble),
                          -471891.9369498712)

    def test_F2(self, hubble):
        assert np.isclose(operand.ParaxialOperand.F2(hubble),
                          0.016864237529848742)

    def test_P1(self, hubble):
        assert np.isclose(operand.ParaxialOperand.P1(hubble),
                          -414291.8559514682)

    def test_P2(self, hubble):
        assert np.isclose(operand.ParaxialOperand.P2(hubble),
                          -57600.06413416607)

    def test_N1(self, hubble):
        assert np.isclose(operand.ParaxialOperand.N1(hubble),
                          -414291.8559514676)

    def test_N2(self, hubble):
        assert np.isclose(operand.ParaxialOperand.N2(hubble),
                          -57600.06413416545)

    def test_EPL(self, hubble):
        assert np.isclose(operand.ParaxialOperand.EPL(hubble), 4910.01016)

    def test_EPD(self, hubble):
        assert np.isclose(operand.ParaxialOperand.EPD(hubble), 2400)

    def test_XPL(self, hubble):
        assert np.isclose(operand.ParaxialOperand.XPL(hubble),
                          -6958.363551642134)

    def test_XPD(self, hubble):
        assert np.isclose(operand.ParaxialOperand.XPD(hubble),
                          289.93210961932573)

    def test_magnification(self, hubble):
        assert np.isclose(operand.ParaxialOperand.magnification(hubble), 0.0)


class TestAberrationOperand:
    def test_seidel(self, hubble):
        assert np.isclose(operand.AberrationOperand.seidels(hubble, 0),
                          0.0014539022855417389)

    def test_TSC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TSC(hubble, 2), 0.0)

    def test_SC(self, hubble):
        assert np.isclose(operand.AberrationOperand.SC(hubble, 2), 0.0)

    def test_CC(self, hubble):
        assert np.isclose(operand.AberrationOperand.CC(hubble, 2), 0.0)

    def test_TCC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TCC(hubble, 1), 0.0)

    def test_TAC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TAC(hubble, 1), 0.0)

    def test_AC(self, hubble):
        assert np.isclose(operand.AberrationOperand.AC(hubble, 1), 0.0)

    def test_TPC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TPC(hubble, 1), 0.0)

    def test_PC(self, hubble):
        assert np.isclose(operand.AberrationOperand.PC(hubble, 1), 0.0)

    def test_DC(self, hubble):
        assert np.isclose(operand.AberrationOperand.DC(hubble, 1), 0.0)

    def test_TAchC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TAchC(hubble, 1), 0.0)

    def test_LchC(self, hubble):
        assert np.isclose(operand.AberrationOperand.LchC(hubble, 1), 0.0)

    def test_TchC(self, hubble):
        assert np.isclose(operand.AberrationOperand.TchC(hubble, 1), 0.0)

    def test_TSC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TSC_sum(hubble), 0.0)

    def test_SC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.SC_sum(hubble), 0.0)

    def test_CC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.CC_sum(hubble), 0.0)

    def test_TCC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TCC_sum(hubble), 0.0)

    def test_TAC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TAC_sum(hubble), 0.0)

    def test_AC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.AC_sum(hubble), 0.0)

    def test_TPC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TPC_sum(hubble), 0.0)

    def test_PC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.PC_sum(hubble), 0.0)

    def test_DC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.DC_sum(hubble),
                          0.03489370392123652)

    def test_TAchC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TAchC_sum(hubble), 0.0)

    def test_LchC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.LchC_sum(hubble), 0.0)

    def test_TchC_sum(self, hubble):
        assert np.isclose(operand.AberrationOperand.TchC_sum(hubble), 0.0)


class TestRayOperand:
    def test_x_intercept(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 1.0, 'Hy': 0.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.x_intercept(**data),
                          -150.42338010762842)

    def test_y_intercept(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.y_intercept(**data),
                          150.42338010762842)

    def test_z_intercept(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.z_intercept(**data),
                          6347.146837237045)

    def test_L(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.L(**data), 0.0)

    def test_M(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.M(**data), 0.021668820646914833)

    def test_N(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'Px': 0.0, 'Py': 0.0, 'wavelength': 0.55}
        assert np.isclose(operand.RayOperand.N(**data), 0.9997652035411982)

    def test_rms_spot_size(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'wavelength': 0.55, 'num_rays': 100}
        assert np.isclose(operand.RayOperand.rms_spot_size(**data),
                          0.025626727777956947)

    def test_rms_spot_size_all_wave(self, hubble):
        data = {'optic': hubble, 'surface_number': -1, 'Hx': 0.0, 'Hy': 1.0,
                'wavelength': 'all', 'num_rays': 100}
        assert np.isclose(operand.RayOperand.rms_spot_size(**data),
                          0.025626727777956947)

    def test_opd_diff(self, hubble):
        data = {'optic': hubble, 'Hx': 0.0, 'Hy': 1.0,
                'wavelength': 0.55, 'num_rays': 3}
        assert np.isclose(operand.RayOperand.OPD_difference(**data),
                          0.0732334055041471)

    def test_opd_diff_on_axis(self, hubble):
        data = {'optic': hubble, 'Hx': 0.0, 'Hy': 0.0,
                'wavelength': 0.55, 'num_rays': 3}
        assert np.isclose(operand.RayOperand.OPD_difference(**data),
                          0.0013295118803701674)

    def test_opd_diff_new_dist(self, hubble):
        data = {'optic': hubble, 'Hx': 0.0, 'Hy': 1.0,
                'wavelength': 0.55, 'num_rays': 100,
                'distribution': 'hexapolar'}
        assert np.isclose(operand.RayOperand.OPD_difference(**data),
                          0.2211995620762635)


class TestOperand:
    def test_get_value(self, hubble):
        input_data = {'optic': hubble}
        op = operand.Operand('f2', 1, 1, input_data)
        assert np.isclose(op.value, 57600.080998403595)

    def test_invalid_operand(self, hubble):
        input_data = {'optic': hubble}
        op = operand.Operand('f3', 1, 1, input_data)
        with pytest.raises(ValueError):
            op.value

    def test_delta(self, hubble):
        input_data = {'optic': hubble}
        op = operand.Operand('f2', 5000, 1, input_data)
        assert np.isclose(op.delta(), 52600.080998403595)

    def test_fun(self, hubble):
        input_data = {'optic': hubble}
        op = operand.Operand('f2', 1e5, 1.5, input_data)
        assert np.isclose(op.fun(), -63599.87850239461)