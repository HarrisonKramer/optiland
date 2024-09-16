from abc import ABC, abstractmethod
import numpy as np
from optiland.optimization.variable import Variable


class BasePerturbation(ABC):
    @abstractmethod
    def sample(self):
        pass  # pragma: no cover


class ScalarPerturbation(BasePerturbation):
    def __init__(self, value):
        self.value = value
        self.size = 1

    def sample(self):
        """Always return the scalar value."""
        return self.value


class RangePerturbation(BasePerturbation):
    def __init__(self, start, end, steps):
        self.values = np.linspace(start, end, steps)
        self.index = 0
        self.size = steps

    def sample(self):
        """Return the next value in the range."""
        if self.index >= len(self.values):
            self.index = 0  # loop over values
        value = self.values[self.index]
        self.index += 1
        return value


class DistributionPerturbation(BasePerturbation):
    def __init__(self, distribution, *params):
        """Initialize a distribution perturbation."""
        self.distribution = distribution
        self.params = params

    def sample(self):
        """Return a random value from the given distribution."""
        if self.distribution == 'normal':
            return np.random.normal(*self.params)
        elif self.distribution == 'uniform':
            return np.random.uniform(*self.params)
        else:
            raise ValueError(f'Unknown distribution: {self.distribution}')


class OpticPerturbation:
    def __init__(self, optic, variable_type, perturbation, **kwargs):
        self.optic = optic
        self.type = variable_type
        self.perturbation = perturbation
        self.variable = Variable(optic, variable_type, apply_scaling=False,
                                 **kwargs)

    def apply(self):
        """Apply the perturbation to the optic."""
        value = self.perturbation.sample()
        self.variable.update_value(value)
