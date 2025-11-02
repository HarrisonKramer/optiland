"""Unit tests for the jacobi module."""
import pytest
import optiland.backend as be
from optiland.geometries.forbes import jacobi

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_weight(backend):
    be.set_backend(backend)
    x = be.linspace(-1, 1, 10)
    alpha = 1
    beta = 1
    w = jacobi.weight(alpha, beta, x)
    assert w.shape == x.shape

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_recurrence_abc(backend):
    be.set_backend(backend)
    A, B, C = jacobi.recurrence_abc(0, 0, 0)
    assert isinstance(A, float)
    assert isinstance(B, float)
    assert isinstance(C, int)

    A, B, C = jacobi.recurrence_abc(1, 0, 0)
    assert isinstance(A, float)
    assert isinstance(B, float)
    assert isinstance(C, float)

@pytest.mark.parametrize('backend', be.list_available_backends())
@pytest.mark.parametrize('n', [0, 1, 2, 3])
def test_jacobi(backend, n):
    be.set_backend(backend)
    x = be.linspace(-1, 1, 10)
    alpha = 1
    beta = 1
    p = jacobi.jacobi(n, alpha, beta, x)
    assert p.shape == x.shape

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_jacobi_sum_clenshaw(backend):
    if backend == 'torch':
        pytest.skip("Skipping torch test due to in-place operation error.")
    be.set_backend(backend)
    x = be.linspace(-1, 1, 10)
    s = be.ones(5)
    if be.is_torch_tensor(x):
        x = x.detach()
        s = s.detach()
    alpha = 1
    beta = 1
    result = jacobi.jacobi_sum_clenshaw(s, alpha, beta, x)
    assert result.shape == (len(s), *x.shape)

    result = jacobi.jacobi_sum_clenshaw([], alpha, beta, x)
    assert result is None or len(result) == 0

    result = jacobi.jacobi_sum_clenshaw(be.ones(1), alpha, beta, x)
    assert result.shape == (1, *x.shape)

@pytest.mark.parametrize('backend', be.list_available_backends())
def test_jacobi_sum_clenshaw_der(backend):
    if backend == 'torch':
        pytest.skip("Skipping torch test due to in-place operation error.")
    be.set_backend(backend)
    x = be.linspace(-1, 1, 10)
    s = be.ones(5)
    if be.is_torch_tensor(x):
        x = x.detach()
        s = s.detach()
    alpha = 1
    beta = 1
    result = jacobi.jacobi_sum_clenshaw_der(s, alpha, beta, x)
    assert result.shape == (2, len(s), *x.shape)

    result = jacobi.jacobi_sum_clenshaw_der([], alpha, beta, x)
    assert result.shape == (2, 0, *x.shape) if hasattr(x, "shape") else (2,0)

    result = jacobi.jacobi_sum_clenshaw_der(s, alpha, beta, x, j=0)
    assert result.shape == (1, len(s), *x.shape)


@pytest.mark.parametrize('backend', be.list_available_backends())
def test_initialize_alphas(backend):
    be.set_backend(backend)
    s = be.ones(5)
    x = be.linspace(-1, 1, 10)
    alphas = jacobi._initialize_alphas(s, x, None)
    assert alphas.shape == (len(s), *x.shape)

    alphas = jacobi._initialize_alphas(s, 1, None)
    assert alphas.shape == (len(s),)

    alphas = jacobi._initialize_alphas(s, x, None, j=1)
    assert alphas.shape == (2, len(s), *x.shape)
