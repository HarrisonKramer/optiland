from copy import deepcopy
from optiland.tolerancing.perturbation import Perturbation
from optiland.tolerancing.compensator import CompensatorOptimizer
from optiland.optimization.operand import Operand


class Tolerancing:
    """
    A class representing a tolerancing problem. This class is the core of the
    tolerancing module. It allows the user to define a tolerancing problem by
    adding operands (metrics), perturbations, and compensators to an optical
    system.

    Args:
        optic: The optic object to be toleranced.
        method: The method to use for optimization. Defaults to 'generic'.
        tol: The tolerance for the compensator optimizer. Defaults to 1e-5.

    Attributes:
        optic: The optic object to be toleranced.
        method: The method to use for optimization. Defaults to 'generic'.
        tol: The tolerance for the compensator optimizer. Defaults to 1e-5.
        _optic_nominal: A deep copy of the optic object representing its
            nominal state.
        operands: A list of operands in the tolerancing problem.
        perturbations: A list of perturbations applied to the optic.
        _compensator: An optimizer for compensating the perturbations.

    Methods:
        __init__(self, optic): Initializes a Tolerancing object with the given
            optic.
        add_operand(self, operand_type, input_data={}): Adds an operand to the
            tolerancing problem.
        add_perturbation(self, variable_type, perturbation, **kwargs): Adds a
            perturbation to the optic.
        add_compensator(self, variable_type, **kwargs): Adds a compensator
            variable to the optimizer.
        apply_compensators(self): Applies compensators to the optic.
        evaluate(self): Evaluates the operands in the tolerancing problem.
        reset(self): Resets the optic to its nominal state.
    """

    def __init__(self, optic, method='generic', tol=1e-5):
        self.optic = optic
        self.method = method
        self.tol = tol
        self._optic_nominal = deepcopy(optic)
        self.operands = []
        self.perturbations = []
        self._compensator = CompensatorOptimizer(method=method, tol=tol)

    def add_operand(self, operand_type: str, input_data: dict = {}):
        """
        Add an operand to the tolerancing problem.

        Args:
            operand_type: The type of the operand.
            input_data: A dictionary of input data for the operand. Defaults to
                an empty dictionary.
        """
        self.operands.append(Operand(operand_type, 0.0, 1.0, input_data))

    def add_perturbation(self, variable_type: str, perturbation: Perturbation,
                         **kwargs):
        """
        Add a perturbation to the optic.

        Args:
            variable_type: The type of the variable to be perturbed, such as
                "radius", "thickness", etc. See the variable module for more
                information.
            perturbation: The perturbation to be applied to the variable.
            **kwargs: Additional keyword arguments for the variable.
        """
        perturbation = Perturbation(self.optic, variable_type,
                                    perturbation, **kwargs)
        self.perturbations.append(perturbation)

    def add_compensator(self, variable_type: str, **kwargs):
        """
        Add a compensator variable to the optimizer.

        Args:
            variable_type: The type of the variable to be used for
                compensation, such as "thickness", etc. See the variable
                module for more information.
            **kwargs: Additional keyword arguments for the variable.
        """
        self._compensator.add_variable(self.optic, variable_type, **kwargs)

    def apply_compensators(self):
        """Apply compensators to the optic."""
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
