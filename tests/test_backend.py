import types
import pytest
from optiland import backend as be


@pytest.fixture(autouse=True)
def reset_backend_state():
    be.set_backend("numpy")
    numpy_backend = be.__getattr__.__globals__["_backends"]["numpy"]
    for attr in list(numpy_backend.__dict__.keys()):
        if attr.startswith("test_attr_"):
            delattr(numpy_backend, attr)
    if hasattr(numpy_backend, "_lib"):
        delattr(numpy_backend, "_lib")
    yield


@pytest.fixture(autouse=True)
def cleanup_dummy_backend():
    yield
    backends = be.__getattr__.__globals__["_backends"]
    if "dummy" in backends:
        del backends["dummy"]


def test_default_backend():
    assert be.get_backend() == "numpy"


def test_list_available_backends():
    assert "numpy" in be.list_available_backends()


def test_invalid_backend():
    with pytest.raises(ValueError):
        be.set_backend("nonexistent")


def test_getattr_nonexistent_attribute():
    with pytest.raises(AttributeError):
        _ = be.some_nonexistent_attribute


def test_getattr_from_backend(monkeypatch):
    fake_backend = types.ModuleType("fake_numpy")
    fake_backend.test_attr_func = lambda x: x * 2
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_func(5) == 10


def test_getattr_from_backend_lib(monkeypatch):
    fake_backend = types.ModuleType("fake_numpy")
    fake_backend._lib = types.SimpleNamespace(test_attr_lib=42)
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_lib == 42


def test_getattr_fallback_priority(monkeypatch):
    fake_backend = types.ModuleType("fake_numpy")
    fake_backend.test_attr_priority = "from_backend"
    fake_backend._lib = types.SimpleNamespace(test_attr_priority="from_lib")
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "numpy", fake_backend)
    monkeypatch.setattr(be, "_current_backend", "numpy")
    assert be.test_attr_priority == "from_backend"


def test_get_backend_after_set(monkeypatch):
    dummy_backend = type("DummyBackend", (), {})()
    setattr(dummy_backend, "dummy_attr", "dummy_value")
    monkeypatch.setitem(be.__getattr__.__globals__["_backends"], "dummy", dummy_backend)
    be.set_backend("dummy")
    assert be.get_backend() == "dummy"
    assert be.dummy_attr == "dummy_value"
