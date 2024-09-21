from optiland.tolerancing.perturbation import BaseSampler, Perturbation
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
        compensator: An optimizer for compensating the perturbations.

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
        self.operands = []
        self.perturbations = []
        self.compensator = CompensatorOptimizer(method=method, tol=tol)

    def add_operand(self, operand_type: str, input_data: dict = {},
                    target: float = None, weight: float = 1.0):
        """
        Add an operand to the tolerancing problem.

        Args:
            operand_type: The type of the operand.
            input_data: A dictionary of input data for the operand. Defaults to
                an empty dictionary.
            target: The target value for the operand. Defaults to None, in
                which case the target is set based on the current value of
                the operand.
            weight: The weight of the operand for optimization during
                compensation. Defaults to 1.0.
        """
        new_operand = Operand(operand_type, target, weight, input_data)
        if target is None:
            new_operand.target = new_operand.value
        self.operands.append(new_operand)

    def add_perturbation(self, variable_type: str, sampler: BaseSampler,
                         **kwargs):
        """
        Add a perturbation to the optic.

        Args:
            variable_type: The type of the variable to be perturbed, such as
                "radius", "thickness", etc. See the variable module for more
                information.
            sampler: The sampler object used to generate perturbation values.
            **kwargs: Additional keyword arguments for the variable.
        """
        perturbation = Perturbation(self.optic, variable_type,
                                    sampler, **kwargs)
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
        self.compensator.add_variable(self.optic, variable_type, **kwargs)

    def apply_compensators(self):
        """Apply compensators to the optic."""
        result = {}
        if self.compensator.has_variables:
            # add operands to the optimization problem (for compensation)
            self.compensator.operands = self.operands

            # run optimizer for compensating the perturbations
            self.compensator.run()

            # record the optimized values
            result = {f'C{i}: {str(var)}': var.value
                      for i, var in enumerate(self.compensator.variables)}
        return result

    def evaluate(self):
        """Evaluate the operands."""
        return [operand.value for operand in self.operands]

    def reset(self):
        """Reset the optic to its initial state."""
        for perturbation in self.perturbations:
            perturbation.reset()

        for compensator in self.compensator.variables:
            compensator.reset()
