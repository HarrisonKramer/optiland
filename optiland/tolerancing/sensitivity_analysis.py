import pandas as pd
import matplotlib.pyplot as plt
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import RangeSampler


class SensitivityAnalysis:

    def __init__(self, tolerancing: Tolerancing):
        self.tolerancing = tolerancing
        self.operand_names = [f'operand_{i}'
                              for i in range(len(self.tolerancing.operands))]
        columns = ['perturbation_id', 'perturbation_value'] + \
            self.operand_names
        self._results = pd.DataFrame(columns=columns)

    def run(self):
        results = []

        for idx, perturbation in enumerate(self.tolerancing.perturbations):
            if not isinstance(perturbation.sampler, RangeSampler):
                raise ValueError('Only range samplers are supported.')

            num_iterations = perturbation.sampler.size
            for _ in range(num_iterations):
                # reset system
                self.tolerancing.reset()

                # apply perturbation
                perturbation.apply()

                # apply compensators
                self.tolerancing.apply_compensators()

                # evaluate operands
                operand_values = self.tolerancing.evaluate()

                # save results
                result = {
                    'perturbation_id': idx,
                    'perturbation_value': perturbation.value,
                }
                result.update({f'operand_{i}': value
                               for i, value in enumerate(operand_values)})
                results.append(result)

        self._results = pd.DataFrame(results)

    def view(self):
        df = self._results
        unique_ids = df['perturbation_id'].unique()
        for idx in unique_ids:
            for name in self.operand_names:
                x = df.loc[df.perturbation_id == idx, 'perturbation_value']
                y = df.loc[df.perturbation_id == idx, name]
                plt.plot(x, y)
                plt.show()
