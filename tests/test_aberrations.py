import optiland.backend as be
import pytest

from optiland.aberrations import Aberrations
from optiland.optic import Optic
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


@pytest.fixture
def simple_singlet():
    """Singlet with single field and wavelength"""
    lens = Optic()

    # add surfaces
    lens.add_surface(index=0, radius=be.inf, thickness=be.inf)
    lens.add_surface(
        index=1,
        thickness=7,
        radius=19.93,
        is_stop=True,
        material="N-SF11",
    )
    lens.add_surface(index=2, thickness=21.48)
    lens.add_surface(index=3)

    # add aperture
    lens.set_aperture(aperture_type="EPD", value=20.0)

    # add field
    lens.set_field_type(field_type="angle")
    lens.add_field(y=0)

    # add wavelength
    lens.add_wavelength(value=0.55, is_primary=True)

    lens.update_paraxial()
    lens.image_solve()
    return lens


class TestDoubleGaussAberrations:
    def test_init(self, double_gauss):
        aberrations = Aberrations(double_gauss)
        assert aberrations.optic == double_gauss

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
        assert be.sum(TSC) == pytest.approx(-0.01964728937767421, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-0.19647289377674193, abs=1e-9)
        assert be.sum(CC) == pytest.approx(0.0019772988166093623, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(0.005931896449828042, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(0.017119527515864978, abs=1e-9)
        assert be.sum(AC) == pytest.approx(0.17119527515864985, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.08132376867613199, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.8132376867613212, abs=1e-9)
        assert be.sum(DC) == pytest.approx(-0.2324205373837797, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(0.0295705512988189, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(0.2957055129881888, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(-0.01804376318260833, abs=1e-9)
        assert S[0] == pytest.approx(-0.003929457875534847, abs=1e-9)
        assert S[1] == pytest.approx(0.0003954597633218682, abs=1e-9)
        assert S[2] == pytest.approx(0.0034239055031729947, abs=1e-9)
        assert S[3] == pytest.approx(-0.016264753735226404, abs=1e-9)
        assert S[4] == pytest.approx(-0.046484107476755930, abs=1e-9)

    def test_third_order_all_functions(self, double_gauss):
        TSC = double_gauss.aberrations.TSC()
        SC = double_gauss.aberrations.SC()
        CC = double_gauss.aberrations.CC()
        TCC = double_gauss.aberrations.TCC()
        TAC = double_gauss.aberrations.TAC()
        AC = double_gauss.aberrations.AC()
        TPC = double_gauss.aberrations.TPC()
        PC = double_gauss.aberrations.PC()
        DC = double_gauss.aberrations.DC()
        TAchC = double_gauss.aberrations.TAchC()
        LchC = double_gauss.aberrations.LchC()
        TchC = double_gauss.aberrations.TchC()

        assert be.sum(TSC) == pytest.approx(-0.01964728937767421, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-0.19647289377674193, abs=1e-9)
        assert be.sum(CC) == pytest.approx(0.0019772988166093623, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(0.005931896449828042, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(0.017119527515864978, abs=1e-9)
        assert be.sum(AC) == pytest.approx(0.17119527515864985, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.08132376867613199, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.8132376867613212, abs=1e-9)
        assert be.sum(DC) == pytest.approx(-0.2324205373837797, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(0.0295705512988189, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(0.2957055129881888, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(-0.01804376318260833, abs=1e-9)


class TestEdmundSinglet:
    def test_init(self, edmund_singlet):
        aberrations = Aberrations(edmund_singlet)
        assert aberrations.optic == edmund_singlet

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
        assert be.sum(TSC) == pytest.approx(-1.7306053598822728, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-3.460883119362552, abs=1e-9)
        assert be.sum(CC) == pytest.approx(0.14252371347566878, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(0.42757114042700617, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(-0.35292203963678487, abs=1e-9)
        assert be.sum(AC) == pytest.approx(-0.7057772717825394, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.2211799550187673, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.4423180410800838, abs=1e-9)
        assert be.sum(DC) == pytest.approx(-0.020852935715656093, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(-0.4947549112756089, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(-0.9894161663592405, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(0.0, abs=1e-9)
        assert S[0] == pytest.approx(-1.730769175588275, abs=1e-9)
        assert S[1] == pytest.approx(0.14253720449059704, abs=1e-9)
        assert S[2] == pytest.approx(-0.352955446544233, abs=1e-9)
        assert S[3] == pytest.approx(-0.22120089147910937, abs=1e-9)
        assert S[4] == pytest.approx(-0.020854909613614383, abs=1e-9)

    def test_third_order_all_functions(self, edmund_singlet):
        TSC = edmund_singlet.aberrations.TSC()
        SC = edmund_singlet.aberrations.SC()
        CC = edmund_singlet.aberrations.CC()
        TCC = edmund_singlet.aberrations.TCC()
        TAC = edmund_singlet.aberrations.TAC()
        AC = edmund_singlet.aberrations.AC()
        TPC = edmund_singlet.aberrations.TPC()
        PC = edmund_singlet.aberrations.PC()
        DC = edmund_singlet.aberrations.DC()
        TAchC = edmund_singlet.aberrations.TAchC()
        LchC = edmund_singlet.aberrations.LchC()
        TchC = edmund_singlet.aberrations.TchC()

        assert be.sum(TSC) == pytest.approx(-1.7306053598822728, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-3.460883119362552, abs=1e-9)
        assert be.sum(CC) == pytest.approx(0.14252371347566878, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(0.42757114042700617, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(-0.35292203963678487, abs=1e-9)
        assert be.sum(AC) == pytest.approx(-0.7057772717825394, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.2211799550187673, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.4423180410800838, abs=1e-9)
        assert be.sum(DC) == pytest.approx(-0.020852935715656093, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(-0.4947549112756089, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(-0.9894161663592405, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(0.0, abs=1e-9)


class TestSingletStopTwo:
    def test_init(self, singlet_stop_surf_two):
        aberrations = Aberrations(singlet_stop_surf_two)
        assert aberrations.optic == singlet_stop_surf_two

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
        assert be.sum(TSC) == pytest.approx(-0.1323786953158293, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-1.074934343302707, abs=1e-9)
        assert be.sum(CC) == pytest.approx(-0.0017810664901602852, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(-0.0053431994704808555, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(-0.0463855041980188, abs=1e-9)
        assert be.sum(AC) == pytest.approx(-0.3766570698925757, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.028095789481046744, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.22814191470407064, abs=1e-9)
        assert be.sum(DC) == pytest.approx(0.006717305986964978, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(-0.2286190741226864, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(-1.8564202776151473, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(0.011126795737552403, abs=1e-9)
        assert S[0] == pytest.approx(-0.0326050034268675, abs=1e-9)
        assert S[1] == pytest.approx(-0.0004386784359568394, abs=1e-9)
        assert S[2] == pytest.approx(-0.01142479550599207, abs=1e-9)
        assert S[3] == pytest.approx(-0.00692002070366785, abs=1e-9)
        assert S[4] == pytest.approx(0.0016544791002946339, abs=1e-9)

    def test_third_order_all_functions(self, singlet_stop_surf_two):
        TSC = singlet_stop_surf_two.aberrations.TSC()
        SC = singlet_stop_surf_two.aberrations.SC()
        CC = singlet_stop_surf_two.aberrations.CC()
        TCC = singlet_stop_surf_two.aberrations.TCC()
        TAC = singlet_stop_surf_two.aberrations.TAC()
        AC = singlet_stop_surf_two.aberrations.AC()
        TPC = singlet_stop_surf_two.aberrations.TPC()
        PC = singlet_stop_surf_two.aberrations.PC()
        DC = singlet_stop_surf_two.aberrations.DC()
        TAchC = singlet_stop_surf_two.aberrations.TAchC()
        LchC = singlet_stop_surf_two.aberrations.LchC()
        TchC = singlet_stop_surf_two.aberrations.TchC()

        assert be.sum(TSC) == pytest.approx(-0.1323786953158293, abs=1e-9)
        assert be.sum(SC) == pytest.approx(-1.074934343302707, abs=1e-9)
        assert be.sum(CC) == pytest.approx(-0.0017810664901602852, abs=1e-9)
        assert be.sum(TCC) == pytest.approx(-0.0053431994704808555, abs=1e-9)
        assert be.sum(TAC) == pytest.approx(-0.0463855041980188, abs=1e-9)
        assert be.sum(AC) == pytest.approx(-0.3766570698925757, abs=1e-9)
        assert be.sum(TPC) == pytest.approx(-0.028095789481046744, abs=1e-9)
        assert be.sum(PC) == pytest.approx(-0.22814191470407064, abs=1e-9)
        assert be.sum(DC) == pytest.approx(0.006717305986964978, abs=1e-9)
        assert be.sum(TAchC) == pytest.approx(-0.2286190741226864, abs=1e-9)
        assert be.sum(LchC) == pytest.approx(-1.8564202776151473, abs=1e-9)
        assert be.sum(TchC) == pytest.approx(0.011126795737552403, abs=1e-9)


class TestSimpleSinglet:
    def test_seidels(self, simple_singlet):
        S = simple_singlet.aberrations.seidels()
        assert S[0] == 0.0
        assert S[1] == 0.0
        assert S[2] == 0.0
        assert S[3] == 0.0
        assert S[4] == 0.0
