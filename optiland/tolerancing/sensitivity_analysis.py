from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import RangePerturbation


@dataclass
class IterationResult:
    perturbation_id: int
    perturbation_value: float
    operand_values: List[float]


class SensitivityAnalysis:

    def __init__(self, tolerancing: Tolerancing):
        self.tolerancing = tolerancing
        self._results = []

    def run(self):
        self._results = []

        for idx, perturbation in enumerate(self.tolerancing.perturbations):
            if not isinstance(perturbation.perturbation, RangePerturbation):
                raise ValueError('Only range perturbations are supported.')

            num_iterations = perturbation.perturbation.size
            for _ in range(num_iterations):
                # reset system
                self.tolerancing.reset()

                # apply perturbation
                perturbation.apply()
                perturbation_value = perturbation.perturbation.sample()

                # apply compensators
                self.tolerancing.apply_compensators()

                # evaluate operands & record results
                operand_values = [operand.value
                                  for operand in self.tolerancing.operands]
                self._results.append(
                    IterationResult(idx, perturbation_value, operand_values)
                    )

    def view(self):
        unique_ids = set(result.perturbation_id for result in self._results)
        for perturbation_id in unique_ids:
            perturbation_values = []
            operand_values = []
            for result in self._results:
                if result.perturbation_id == perturbation_id:
                    perturbation_values.append(result.perturbation_value)
                    operand_values.append(result.operand_values)
            plt.scatter(perturbation_values, operand_values,
                        label=f'Perturbation ID: {perturbation_id}', s=4)
        plt.xlabel('Perturbation Value')
        plt.ylabel('Operand Value')
        plt.legend()
        plt.show()
