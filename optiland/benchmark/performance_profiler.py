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
    and saves the results in a single consolidated HTML report.
    """
    # cfg
    benchmark_systems = [
        AsphericSinglet,
        MobileImagingSystem,
        CatadiopticMicroscope,
        ToroidalSinglet,
        BeamShaperAsphere,
        BeamShaperForbesQbfs,
    ]
    backends = ["numpy", "torch"]
    num_rays = 100  # we can change this later but 100k is a good start imo
    output_dir = "optiland/benchmark/benchmarking_results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # setup
    # create a single profiler to capture all sessions
    profiler = Profiler()
    session_map = {}
    session_count = 0

    # bechmarking loop
    for system_class in benchmark_systems:
        for backend_name in backends:
            session_name = f"{system_class.__name__} ({backend_name})"
            print(f"Benchmarking {session_name}...")

            session_count += 1
            session_map[session_count] = session_name

            # setup optic and backend
            backend.set_backend(backend_name)
            optic = system_class.create_system()

            # profiling
            profiler.start()
            run_ray_trace(optic, num_rays)
            profiler.stop()

    # reporting stuff
    print("\n Benchmarking complete. Generating combined report...")
    report_path = os.path.join(output_dir, "combined_performance_profile.html")

    print("Session Mapping:")
    for count, name in session_map.items():
        print(f"  Session {count}: {name}")

    print("\n   Opening combined report in browser...")
    profiler.open_in_browser()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(profiler.output_html())
    print(f"Combined report saved to {report_path}\n")


if __name__ == "__main__":
    run_benchmarking()
