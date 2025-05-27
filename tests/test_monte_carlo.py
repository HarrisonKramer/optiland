from unittest.mock import patch

import matplotlib
import matplotlib.pyplot as plt
import pytest

from optiland.samples.objectives import ReverseTelephoto
from optiland.tolerancing.core import Tolerancing
from optiland.tolerancing.monte_carlo import MonteCarlo
from optiland.tolerancing.perturbation import DistributionSampler

matplotlib.use("Agg")  # use non-interactive backend for testing


@pytest.fixture
def monte_carlo():
    optic = ReverseTelephoto()
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand(operand_type="f1", input_data={"optic": optic})
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})
    sampler = DistributionSampler("normal", loc=100, scale=2)
    tolerancing.add_perturbation("radius", sampler, surface_number=1)
    tolerancing.add_compensator("thickness", surface_number=2)
    monte_carlo = MonteCarlo(tolerancing)
    return monte_carlo


@pytest.fixture
def monte_carlo_no_compensator():
    optic = ReverseTelephoto()
    tolerancing = Tolerancing(optic)
    tolerancing.add_operand(operand_type="f1", input_data={"optic": optic})
    tolerancing.add_operand(operand_type="f2", input_data={"optic": optic})
    sampler = DistributionSampler("normal", loc=100, scale=2)
    tolerancing.add_perturbation("radius", sampler, surface_number=1)
    monte_carlo = MonteCarlo(tolerancing)
    return monte_carlo


def test_run(monte_carlo):
    num_iterations = 10
    monte_carlo.run(num_iterations)

    # Check if the results DataFrame has the correct shape
    assert len(monte_carlo._results) == num_iterations

    res = {
        "0: f1",
        "1: f2",
        "Radius of Curvature, Surface 1",
        "C0: Thickness, Surface 2",
    }
    assert set(monte_carlo._results.columns) == res


def test_run_no_compensator(monte_carlo_no_compensator):
    num_iterations = 10
    monte_carlo_no_compensator.run(num_iterations)

    # Check if the results DataFrame has the correct shape
    assert len(monte_carlo_no_compensator._results) == num_iterations

    res = {
        "0: f1",
        "1: f2",
        "Radius of Curvature, Surface 1",
    }
    assert set(monte_carlo_no_compensator._results.columns) == res


@patch("matplotlib.pyplot.show")
def test_view_histogram(mock_show, monte_carlo_no_compensator):
    monte_carlo_no_compensator.run(10)
    monte_carlo_no_compensator.view_histogram(kde=True)
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_histogram_no_kde(mock_show, monte_carlo_no_compensator):
    monte_carlo_no_compensator.run(10)
    monte_carlo_no_compensator.view_histogram(kde=False)
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_cdf(mock_show, monte_carlo_no_compensator):
    monte_carlo_no_compensator.run(10)
    monte_carlo_no_compensator.view_cdf()
    mock_show.assert_called_once()
    plt.close()


@patch("matplotlib.pyplot.show")
def test_view_heatmap(mock_show, monte_carlo_no_compensator):
    monte_carlo_no_compensator.run(10)
    monte_carlo_no_compensator.view_heatmap(figsize=(8, 6))
    mock_show.assert_called_once()
    plt.close()


def test_invalid_plot_type(monte_carlo):
    msg = "Invalid plot type: invalid"
    with pytest.raises(ValueError, match=msg):
        monte_carlo._plot(plot_type="invalid")


def test_validate_no_operands(monte_carlo):
    monte_carlo.tolerancing.operands = []
    msg = "No operands found in the tolerancing system."
    with pytest.raises(ValueError, match=msg):
        monte_carlo._validate()


def test_validate_no_perturbations(monte_carlo):
    monte_carlo.tolerancing.perturbations = []
    msg = "No perturbations found in the tolerancing system."
    with pytest.raises(ValueError, match=msg):
        monte_carlo._validate()
