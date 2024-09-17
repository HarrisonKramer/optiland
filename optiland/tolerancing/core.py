from copy import deepcopy
from optiland.tolerancing.perturbation import Perturbation
from optiland.tolerancing.compensator import CompensatorOptimizer
from optiland.optimization.operand import Operand


class Tolerancing:

    def __init__(self, optic):
        self.optic = optic
        self._optic_nominal = deepcopy(optic)
        self.operands = []
        self.perturbations = []
        self._compensator = CompensatorOptimizer()

    def add_operand(self, operand_type, input_data={}):
        """Add an operand to the tolerancing problem."""
        self.operands.append(Operand(operand_type, 0.0, 1.0, input_data))

    def add_perturbation(self, variable_type, perturbation, **kwargs):
        perturbation = Perturbation(self.optic, variable_type,
                                    perturbation, **kwargs)
        self.perturbations.append(perturbation)

    def add_compensator(self, variable_type, **kwargs):
        self._compensator.add_variable(self.optic, variable_type, **kwargs)

    def apply_compensators(self):
        if self._compensator.has_variables:
            # add operands to the optimization problem (for compensation)
            self._compensator.operands = self.operands

            # Run optimizer for compensating the perturbations
            self._compensator.run()

    def evaluate(self):
        """Evaluate the operands."""
        return [operand.value for operand in self.operands]

    def reset(self):
        """Reset the optic to its nominal state."""
        self.optic = deepcopy(self._optic_nominal)
