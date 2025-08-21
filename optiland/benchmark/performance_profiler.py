from __future__ import annotations

import os

from pyinstrument import Profiler

from optiland import backend
from optiland.benchmark.benchmarking_systems import (
    AsphericSinglet,
    BeamShaperAsphere,
    BeamShaperForbesQbfs,
    CatadiopticMicroscope,
    MobileImagingSystem,
    ToroidalSinglet,
    run_ray_trace,
)


def run_benchmarking():
    """
    Runs the ray tracing benchmark for all defined optical systems and backends,
    and saves a separate HTML report for each test.
    """
    # --- Configuration ---
    benchmark_systems = [
        AsphericSinglet,
        MobileImagingSystem,
        CatadiopticMicroscope,
        ToroidalSinglet,
        BeamShaperAsphere,
        BeamShaperForbesQbfs,
    ]
    backends = ["numpy", "torch"]
    num_rays = 10000  # we can change this but 10k is a good start imo
    output_dir = "optiland/benchmark/benchmarking_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # benchmarking loop
    for system_class in benchmark_systems:
        for backend_name in backends:
            session_name = f"{system_class.__name__}_{backend_name}"
            print(f"Benchmarking {session_name}...")

            # optic and backend setup
            backend.set_backend(backend_name)
            optic = system_class.create_system()

            # create a new profiler for each session
            profiler = Profiler()

            # profiling
            profiler.start()
            run_ray_trace(optic, num_rays)
            profiler.stop()

            # reports
            report_name = f"{session_name}_profile.html"
            report_path = os.path.join(output_dir, report_name)

            print(f"   Opening report for {session_name}...")
            profiler.open_in_browser()

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(profiler.output_html())
            print(f"Report saved to {report_path}\n")
    print("\n Benchmarking complete.")


if __name__ == "__main__":
    run_benchmarking()
