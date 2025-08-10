def decorator_return_as_is(*args, **kwargs):
    def f(x):
        return x

    if len(args) == 0:
        return f

    if callable(args[0]):
        return args[0]

    return f


jit = decorator_return_as_is
njit = decorator_return_as_is
prange = range
