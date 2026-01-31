import os
import pytest
from optiland.samples.objectives import CookeTriplet
from optiland.reporting.engine import StandardPerformanceReport
from optiland.reporting.metrics import FirstOrderMetrics, ImageQualityMetrics, WavefrontMetrics, SeidelMetrics

def test_metrics_calculation():
    optic = CookeTriplet()

    fo = FirstOrderMetrics.calculate(optic)
    assert "EFL" in fo
    assert fo["EFL"] > 0

    iq = ImageQualityMetrics.calculate(optic)
    assert len(iq) > 0

    wf = WavefrontMetrics.calculate(optic)
    assert len(wf) > 0

    seidel = SeidelMetrics.calculate(optic)
    assert "Spherical (S1)" in seidel

def test_report_generation(tmp_path):
    optic = CookeTriplet()
    report = StandardPerformanceReport(optic)

    filename = tmp_path / "test_report.pdf"
    report.save(str(filename))

    assert filename.exists()
    assert filename.stat().st_size > 0
