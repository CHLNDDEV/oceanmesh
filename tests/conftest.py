import matplotlib
import matplotlib.pyplot as plt
import pytest


matplotlib.use("Agg")  # non-interactive backend


@pytest.fixture(autouse=True)
def _suppress_matplotlib_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
