import optiland.backend as be
import pytest

from optiland.samples.simple import Edmund_49_847
from optiland.tolerancing.compensator import CompensatorOptimizer
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import Perturbation, ScalarSampler


@pytest.fixture
def setup_tolerancing():
    optic = Edmund_49_847()
    tolerancing = Tolerancing(optic)
    return tolerancing, optic


def test_init(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    assert tolerancing.optic == optic
    assert tolerancing.method == "generic"
    assert tolerancing.tol == 1e-5
    assert tolerancing.operands == []
    assert tolerancing.perturbations == []
    assert isinstance(tolerancing.compensator, CompensatorOptimizer)


def test_add_operand(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    operand_type = "f2"
    input_data = {"optic": optic}
    target = 20.0
    weight = 2.0

    tolerancing.add_operand(operand_type, input_data, target, weight)
    assert len(tolerancing.operands) == 1
    operand = tolerancing.operands[0]
    assert operand.operand_type == operand_type
    assert operand.input_data == input_data
    assert operand.target == target
    assert operand.weight == weight


def test_add_operand_no_target(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    operand_type = "f2"
    input_data = {"optic": optic}
    weight = 2.0
    target = optic.paraxial.f2()

    tolerancing.add_operand(
        operand_type=operand_type,
        input_data=input_data,
        weight=weight,
    )
    assert len(tolerancing.operands) == 1
    operand = tolerancing.operands[0]
    assert operand.operand_type == operand_type
    assert operand.input_data == input_data
    assert be.isclose(operand.target, target)
    assert operand.weight == weight


def test_add_perturbation(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    variable_type = "radius"
    perturbation = ScalarSampler(value=100.0)

    tolerancing.add_perturbation(variable_type, perturbation, surface_number=1)
    assert len(tolerancing.perturbations) == 1
    added_perturbation = tolerancing.perturbations[0]
    assert isinstance(added_perturbation, Perturbation)
    assert added_perturbation.optic == optic
    assert added_perturbation.type == variable_type


def test_add_compensator(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    variable_type = "thickness"

    tolerancing.add_compensator(variable_type, surface_number=2)
    assert len(tolerancing.compensator.variables) == 1
    compensator_variable = tolerancing.compensator.variables[0]
    assert compensator_variable.optic == optic
    assert compensator_variable.type == variable_type


def test_apply_compensators(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_compensator("radius", surface_number=1)
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})

    result = tolerancing.apply_compensators()
    first_key = list(result.keys())[0]
    assert first_key == "C0: Radius of Curvature, Surface 1"


def test_evaluate(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_operand(operand_type="f1", input_data={"optic": optic})
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})

    result = tolerancing.evaluate()
    assert be.allclose(result, [optic.paraxial.f1(), optic.paraxial.f2()])


def test_reset(setup_tolerancing):
    tolerancing, optic = setup_tolerancing
    tolerancing.add_perturbation("radius", ScalarSampler(value=100.0), surface_number=1)
    tolerancing.perturbations[0].apply()
    assert tolerancing.perturbations[0].value == 100.0
    tolerancing.reset()
    assert tolerancing.perturbations[0].value == 19.93  # original value
