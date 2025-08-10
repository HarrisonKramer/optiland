def __getattr__(name):
    raise RuntimeError(
        f"VTK functionality is not available in this environment. Calling: {name}"
    )
