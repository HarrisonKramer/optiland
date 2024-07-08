import pytest
import numpy as np
from optiland.samples.objectives import DoubleGauss
from optiland.samples.simple import Edmund_49_847, SingletStopSurf2


@pytest.fixture
def double_gauss():
    return DoubleGauss()


@pytest.fixture
def edmund_singlet():
    return Edmund_49_847()


@pytest.fixture
def singlet_stop_surf_two():
    return SingletStopSurf2()


class TestDoubleGaussAberrations:
    def test_seidels(self, double_gauss):
        S = double_gauss.aberrations.seidels()
        assert S[0] == pytest.approx(-0.003929457875534847, abs=1e-9)
        assert S[1] == pytest.approx(0.0003954597633218682, abs=1e-9)
        assert S[2] == pytest.approx(0.0034239055031729947, abs=1e-9)
        assert S[3] == pytest.approx(-0.016264753735226404, abs=1e-9)
        assert S[4] == pytest.approx(-0.046484107476755930, abs=1e-9)

    def test_third_order(self, double_gauss):
        data = double_gauss.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert np.sum(TSC) == pytest.approx(-0.01964728937767421, abs=1e-9)
        assert np.sum(SC) == pytest.approx(-0.19647289377674193, abs=1e-9)
        assert np.sum(CC) == pytest.approx(0.0019772988166093623, abs=1e-9)
        assert np.sum(TCC) == pytest.approx(0.005931896449828042, abs=1e-9)
        assert np.sum(TAC) == pytest.approx(0.017119527515864978, abs=1e-9)
        assert np.sum(AC) == pytest.approx(0.17119527515864985, abs=1e-9)
        assert np.sum(TPC) == pytest.approx(-0.08132376867613199, abs=1e-9)
        assert np.sum(PC) == pytest.approx(-0.8132376867613212, abs=1e-9)
        assert np.sum(DC) == pytest.approx(-0.2324205373837797, abs=1e-9)
        assert np.sum(TAchC) == pytest.approx(0.0295705512988189, abs=1e-9)
        assert np.sum(LchC) == pytest.approx(0.2957055129881888, abs=1e-9)
        assert np.sum(TchC) == pytest.approx(-0.01804376318260833, abs=1e-9)
        assert S[0] == pytest.approx(-0.003929457875534847, abs=1e-9)
        assert S[1] == pytest.approx(0.0003954597633218682, abs=1e-9)
        assert S[2] == pytest.approx(0.0034239055031729947, abs=1e-9)
        assert S[3] == pytest.approx(-0.016264753735226404, abs=1e-9)
        assert S[4] == pytest.approx(-0.046484107476755930, abs=1e-9)


class TestEdmundSinglet:
    def test_seidels(self, edmund_singlet):
        S = edmund_singlet.aberrations.seidels()
        assert S[0] == pytest.approx(-1.730769175588275, abs=1e-9)
        assert S[1] == pytest.approx(0.14253720449059704, abs=1e-9)
        assert S[2] == pytest.approx(-0.352955446544233, abs=1e-9)
        assert S[3] == pytest.approx(-0.22120089147910937, abs=1e-9)
        assert S[4] == pytest.approx(-0.020854909613614383, abs=1e-9)

    def test_third_order(self, edmund_singlet):
        data = edmund_singlet.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert np.sum(TSC) == pytest.approx(-1.7306053598822728, abs=1e-9)
        assert np.sum(SC) == pytest.approx(-3.460883119362552, abs=1e-9)
        assert np.sum(CC) == pytest.approx(0.14252371347566878, abs=1e-9)
        assert np.sum(TCC) == pytest.approx(0.42757114042700617, abs=1e-9)
        assert np.sum(TAC) == pytest.approx(-0.35292203963678487, abs=1e-9)
        assert np.sum(AC) == pytest.approx(-0.7057772717825394, abs=1e-9)
        assert np.sum(TPC) == pytest.approx(-0.2211799550187673, abs=1e-9)
        assert np.sum(PC) == pytest.approx(-0.4423180410800838, abs=1e-9)
        assert np.sum(DC) == pytest.approx(-0.020852935715656093, abs=1e-9)
        assert np.sum(TAchC) == pytest.approx(-0.4947549112756089, abs=1e-9)
        assert np.sum(LchC) == pytest.approx(-0.9894161663592405, abs=1e-9)
        assert np.sum(TchC) == pytest.approx(0.0, abs=1e-9)
        assert S[0] == pytest.approx(-1.730769175588275, abs=1e-9)
        assert S[1] == pytest.approx(0.14253720449059704, abs=1e-9)
        assert S[2] == pytest.approx(-0.352955446544233, abs=1e-9)
        assert S[3] == pytest.approx(-0.22120089147910937, abs=1e-9)
        assert S[4] == pytest.approx(-0.020854909613614383, abs=1e-9)


class TestSingletStopTwo:
    def test_seidels(self, singlet_stop_surf_two):
        S = singlet_stop_surf_two.aberrations.seidels()
        assert S[0] == pytest.approx(-0.0326050034268675, abs=1e-9)
        assert S[1] == pytest.approx(-0.0004386784359568394, abs=1e-9)
        assert S[2] == pytest.approx(-0.01142479550599207, abs=1e-9)
        assert S[3] == pytest.approx(-0.00692002070366785, abs=1e-9)
        assert S[4] == pytest.approx(0.0016544791002946339, abs=1e-9)

    def test_third_order(self, singlet_stop_surf_two):
        data = singlet_stop_surf_two.aberrations.third_order()
        TSC, SC, CC, TCC, TAC, AC, TPC, PC, DC, TAchC, LchC, TchC, S = data
        assert np.sum(TSC) == pytest.approx(-0.1323786953158293, abs=1e-9)
        assert np.sum(SC) == pytest.approx(-1.074934343302707, abs=1e-9)
        assert np.sum(CC) == pytest.approx(-0.0017810664901602852, abs=1e-9)
        assert np.sum(TCC) == pytest.approx(-0.0053431994704808555, abs=1e-9)
        assert np.sum(TAC) == pytest.approx(-0.0463855041980188, abs=1e-9)
        assert np.sum(AC) == pytest.approx(-0.3766570698925757, abs=1e-9)
        assert np.sum(TPC) == pytest.approx(-0.028095789481046744, abs=1e-9)
        assert np.sum(PC) == pytest.approx(-0.22814191470407064, abs=1e-9)
        assert np.sum(DC) == pytest.approx(0.006717305986964978, abs=1e-9)
        assert np.sum(TAchC) == pytest.approx(-0.2286190741226864, abs=1e-9)
        assert np.sum(LchC) == pytest.approx(-1.8564202776151473, abs=1e-9)
        assert np.sum(TchC) == pytest.approx(0.011126795737552403, abs=1e-9)
        assert S[0] == pytest.approx(-0.0326050034268675, abs=1e-9)
        assert S[1] == pytest.approx(-0.0004386784359568394, abs=1e-9)
        assert S[2] == pytest.approx(-0.01142479550599207, abs=1e-9)
        assert S[3] == pytest.approx(-0.00692002070366785, abs=1e-9)
        assert S[4] == pytest.approx(0.0016544791002946339, abs=1e-9)
