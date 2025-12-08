import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import pytest

@pytest.fixture(autouse=True)
def _suppress_matplotlib_show(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)