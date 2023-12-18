import abc
import numpy as np
import matplotlib.pyplot as plt


class BaseDistribution:

    @abc.abstractmethod
    def generate_points(self, num_points: int):
        pass

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def dy(self):
        return self.y[1] - self.y[0]

    def view(self):
        plt.plot(self.x, self.y, 'k*')
        t = np.linspace(0, 2 * np.pi, 256)
        x, y = np.cos(t), np.sin(t)
        plt.plot(x, y, 'r')
        plt.xlabel('Normalized Pupil Coordinate X')
        plt.ylabel('Normalized Pupil Coordinate Y')
        plt.axis('equal')
        plt.show()


class LineXDistribution(BaseDistribution):

    def __init__(self, positive_only: bool = False):
        self.positive_only = positive_only

    def generate_points(self, num_points: int):
        if self.positive_only:
            self.x = np.linspace(0, 1, num_points)
        else:
            self.x = np.linspace(-1, 1, num_points)
        self.y = np.zeros(num_points)


class LineYDistribution(BaseDistribution):

    def __init__(self, positive_only: bool = False):
        self.positive_only = positive_only

    def generate_points(self, num_points: int):
        self.x = np.zeros(num_points)
        if self.positive_only:
            self.y = np.linspace(0, 1, num_points)
        else:
            self.y = np.linspace(-1, 1, num_points)


class RandomDistribution(BaseDistribution):

    def generate_points(self, num_points: int):
        r = np.random.rand(num_points)
        theta = np.random.rand(num_points) * 2 * np.pi
        self.x = np.sqrt(r) * np.cos(theta)
        self.y = np.sqrt(r) * np.sin(theta)


class UniformDistribution(BaseDistribution):

    def generate_points(self, num_points: int):
        x = np.linspace(-1, 1, num_points)
        x, y = np.meshgrid(x, x)
        x = x.ravel()
        y = y.ravel()
        R = np.sqrt(x**2 + y**2)
        self.x = x[R <= 1]
        self.y = y[R <= 1]


class HexagonalDistribution(BaseDistribution):

    def generate_points(self, num_rings: int = 6):
        x = [0]
        y = [0]
        r = np.linspace(0, 1, num_rings+1)
        for i in range(num_rings):
            num_theta = 6 * (i + 1)
            theta = np.linspace(0, 2*np.pi, num_theta+1)
            for j in range(num_theta):
                x.append(r[i+1] * np.cos(theta[j]))
                y.append(r[i+1] * np.sin(theta[j]))

        self.x = np.array(x)
        self.y = np.array(y)


class CrossDistribution(BaseDistribution):

    def generate_points(self, num_points: int):
        x1 = np.zeros(num_points)
        x2 = np.linspace(-1, 1, num_points)
        y1 = np.linspace(-1, 1, num_points)
        y2 = np.zeros(num_points)
        self.x = np.concatenate((x1, x2))
        self.y = np.concatenate((y1, y2))


def create_distribution(distribution_type):
    if distribution_type == 'line_x':
        return LineXDistribution()
    elif distribution_type == 'line_y':
        return LineYDistribution()
    elif distribution_type == 'positive_line_x':
        return LineXDistribution(positive_only=True)
    elif distribution_type == 'positive_line_y':
        return LineYDistribution(positive_only=True)
    elif distribution_type == 'random':
        return RandomDistribution()
    elif distribution_type == 'uniform':
        return UniformDistribution()
    elif distribution_type == 'hexapolar':
        return HexagonalDistribution()
    elif distribution_type == 'cross':
        return CrossDistribution()
    else:
        raise ValueError('Invalid distribution type')
