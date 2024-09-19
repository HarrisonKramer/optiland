import pandas as pd
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.sensitivity_analysis import SensitivityAnalysis


class MonteCarlo(SensitivityAnalysis):

    def __init__(self, tolerancing: Tolerancing):
        super().__init__(tolerancing)

    def run(self, num_iterations: int):
        results = []

        for _ in range(num_iterations):
            # reset the tolerancing system
            self.tolerancing.reset()

            # apply perturbations
            for perturbation in self.tolerancing.perturbations:
                perturbation.apply()

            # apply compensators
            compensator_result = self.tolerancing.apply_compensators()

            # evaluate operands
            operand_values = self.tolerancing.evaluate()

            # save results
            result = {}

            # save results - perturbation type & value
            for perturbation in self.tolerancing.perturbations:
                key = str(perturbation.variable)
                result[key] = perturbation.value

            # save results - operand values
            result.update({
                f'{name}': value
                for name, value in zip(self.operand_names, operand_values)
            })

            # save results - compensator values
            result.update(compensator_result)

            results.append(result)

        self._results = pd.DataFrame(results)

    def view_histogram(self):
        pass

    def view_cdf(self):
        pass

    def view_heatmap(self):
        pass

    def _validate(self):
        """
        Validates the tolerancing system before performing Monte Carlo
        analysis.

        Raises:
            ValueError: If no operands are found in the tolerancing system.
            ValueError: If no perturbations are found in the tolerancing
                system.
        """
        if not self.tolerancing.operands:
            raise ValueError('No operands found in the tolerancing system.')

        if not self.tolerancing.perturbations:
            raise ValueError('No perturbations found in the tolerancing '
                             'system.')
