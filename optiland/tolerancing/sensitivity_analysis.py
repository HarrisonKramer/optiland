"""Sensitivity Analysis Module

This module contains the SensitivityAnalysis class for performing sensitivity
analysis on a tolerancing system. The SensitivityAnalysis class allows users to
run a sensitivity analysis on a Tolerancing object and visualize the results.

Kramer Harrison, 2024
"""

import matplotlib.pyplot as plt
import pandas as pd

import optiland.backend as be
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.perturbation import RangeSampler


class SensitivityAnalysis:
    """Class for performing sensitivity analysis on a tolerancing system.

    Args:
        tolerancing (Tolerancing): The tolerancing system to perform
            sensitivity analysis on.

    Attributes:
        tolerancing (Tolerancing): The tolerancing system to perform
            sensitivity analysis on.
        operand_names (list): List of operand names in the tolerancing system.
        _results (pd.DataFrame): DataFrame to store the sensitivity analysis
            results.

    Methods:
        run(): Runs the sensitivity analysis.
        get_results(): Returns the sensitivity analysis results.
        view(figsize=(2.2, 3), sharex='col', sharey='row'): Visualizes the
            sensitivity analysis results.

    """

    def __init__(self, tolerancing: Tolerancing):
        self.tolerancing = tolerancing
        self.operand_names = [
            f"{i}: {operand}" for i, operand in enumerate(tolerancing.operands)
        ]
        self._results = pd.DataFrame()
        self._validate()

    def run(self):
        """Run the sensitivity analysis.

        This method performs a sensitivity analysis by iterating over the
        perturbations defined in the tolerancing object. For each
        perturbation, it applies the perturbation, applies compensators,
        evaluates operands, and saves the results. The results are stored in a
        pandas DataFrame in the _results attribute.

        Raises:
            ValueError: If a perturbation sampler other than RangeSampler is
                used.

        """
        results = []

        for perturbation in self.tolerancing.perturbations:
            if not isinstance(perturbation.sampler, RangeSampler):
                raise ValueError("Only range samplers are supported.")

            num_iterations = perturbation.sampler.size
            for _ in range(num_iterations):
                # reset system
                self.tolerancing.reset()

                # apply perturbation
                perturbation.apply()

                # apply compensators & save results
                compensator_result = self.tolerancing.apply_compensators()

                # evaluate operands
                operand_values = self.tolerancing.evaluate()

                # save results - perturbation type & value
                result = {
                    "perturbation_type": str(perturbation.variable),
                    "perturbation_value": perturbation.value,
                }

                # save results - operand values
                result.update(
                    {
                        f"{name}": value
                        for name, value in zip(self.operand_names, operand_values)
                    },
                )

                # save results - compensator values
                result.update(compensator_result)

                results.append(result)

        self._results = pd.DataFrame(results)
        self.tolerancing.reset()

    def get_results(self):
        """Returns the results of the sensitivity analysis.

        Returns:
            pd.DataFrame: The results of the sensitivity analysis.

        """
        return self._results

    def view(self, figsize=(2.5, 3.3), sharex="col", sharey="row"):
        """Visualizes the sensitivity analysis results.

        Args:
            figsize (tuple, optional): The size of the figure in inches for
                each subplot. Default is (2.2, 3).
            sharex (str, optional): Specifies how the x-axis is shared among
                subplots. Default is 'col'.
            sharey (str, optional): Specifies how the y-axis is shared among
                subplots. Default is 'row'.

        """
        df = self._results
        unique_types = df["perturbation_type"].unique()

        m = len(self.operand_names)
        n = len(unique_types)

        size_x = m * figsize[0]
        size_y = n * figsize[1]

        fig, axes = plt.subplots(
            m,
            n,
            figsize=(size_y, size_x),
            sharex=sharex,
            sharey=sharey,
        )

        # handle single row and/or column
        axes = be.array(axes).reshape(m, n)

        for i, name in enumerate(self.operand_names):
            for j, pert_type in enumerate(unique_types):
                x = df.loc[df.perturbation_type == pert_type, "perturbation_value"]
                y = df.loc[df.perturbation_type == pert_type, name]

                axes[i, j].plot(x, y, color=f"C{i}", linewidth=2)
                axes[i, j].grid()

                if j == 0:
                    axes[i, j].set_ylabel(name)

                if i == m - 1:
                    axes[i, j].set_xlabel(pert_type)

        plt.tight_layout()
        plt.show()

    def _validate(self):
        """Validates the tolerancing system before performing sensitivity
        analysis.

        Raises:
            ValueError: If no operands are found in the tolerancing system.
            ValueError: If no perturbations are found in the tolerancing
                system.
            ValueError: If the number of operands exceeds 6.
            ValueError: If the number of perturbations exceeds 6.

        """
        if not self.tolerancing.operands:
            raise ValueError("No operands found in the tolerancing system.")

        if not self.tolerancing.perturbations:
            raise ValueError("No perturbations found in the tolerancing system.")

        if len(self.tolerancing.operands) > 6:
            raise ValueError("Sensitivity analysis is limited to 6 operands.")

        if len(self.tolerancing.perturbations) > 6:
            raise ValueError("Sensitivity analysis is limited to 6 perturbations.")
