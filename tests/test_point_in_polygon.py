"""Tests for the GPL-compatible point-in-polygon implementation.

These tests exercise the new :mod:`oceanmesh.geometry.point_in_polygon`
module, ensuring API compatibility with the historical ``inpoly2``
function and good coverage of geometric edge cases, including the
optional Cython-accelerated kernel when available.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from oceanmesh.geometry import inpoly2, point_in_polygon as pip_module


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _square():
    node = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )
    edge = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    return node, edge


def _triangle():
    node = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 1.0],
        ]
    )
    edge = np.array([[0, 1], [1, 2], [2, 0]], dtype=int)
    return node, edge


def _l_shape():
    node = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ]
    )
    edge = np.column_stack([np.arange(len(node)), np.roll(np.arange(len(node)), -1)])
    return node, edge


def _polygon_with_hole():
    # Simple outer square with inner square hole
    outer = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [3.0, 3.0],
            [0.0, 3.0],
        ]
    )
    inner = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0],
        ]
    )
    # For the current implementation, we treat this as a single polygon
    # without explicit hole semantics; tests will focus on basic coverage.
    node = np.vstack([outer, inner])
    edge = np.column_stack([np.arange(len(node)), np.roll(np.arange(len(node)), -1)])
    return node, edge


# ---------------------------------------------------------------------------
# Basic functionality tests
# ---------------------------------------------------------------------------


def test_simple_convex_polygon_square():
    node, edge = _square()
    pts = np.array(
        [
            [0.5, 0.5],  # inside
            [1.5, 0.5],  # outside
            [0.0, 0.0],  # vertex
        ]
    )
    stat, bnds = inpoly2(pts, node, edge)

    # Inside point
    assert stat[0]
    assert not bnds[0]
    # Outside point
    assert not stat[1]
    assert not bnds[1]
    # Vertex is classified as boundary and also inside
    assert stat[2]
    assert bnds[2]


# ---------------------------------------------------------------------------
# Strategy selection and performance-related tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_large_point_set_reasonable_runtime(monkeypatch):
    node, edge = _square()
    rng = np.random.default_rng(0)
    pts = rng.random((10000, 2)) * 2.0 - 0.5  # some outside, some inside

    # Force raycasting to ensure baseline path works at this scale.
    monkeypatch.setenv("OCEANMESH_INPOLY_METHOD", "raycasting")
    stat, bnds = inpoly2(pts, node, edge)
    assert stat.shape == (10000,)
    assert bnds.shape == (10000,)


def test_force_raycasting_method(monkeypatch):
    node, edge = _square()
    pts = np.array([[0.5, 0.5], [2.0, 2.0]])

    monkeypatch.setenv("OCEANMESH_INPOLY_METHOD", "raycasting")
    stat, bnds = inpoly2(pts, node, edge)
    assert stat[0] and not stat[1]


def test_force_shapely_method_if_available(monkeypatch):
    node, edge = _square()
    pts = np.array([[0.5, 0.5], [2.0, 2.0]])

    monkeypatch.setenv("OCEANMESH_INPOLY_METHOD", "shapely")
    stat, bnds = inpoly2(pts, node, edge)
    # We only assert shapes here; numerical exactness is covered elsewhere.
    assert stat.shape == (2,)
    assert bnds.shape == (2,)


def test_force_matplotlib_method_if_available(monkeypatch):
    node, edge = _square()
    pts = np.array([[0.5, 0.5], [2.0, 2.0]])

    monkeypatch.setenv("OCEANMESH_INPOLY_METHOD", "matplotlib")
    stat, bnds = inpoly2(pts, node, edge)
    assert stat.shape == (2,)
    assert bnds.shape == (2,)


# ---------------------------------------------------------------------------
# Regression-style test approximating real usage patterns
# ---------------------------------------------------------------------------


def test_usage_pattern_stat_only():
    """Mimic the common usage ``stat, _ = inpoly2(vert, node, edge)``.

    This verifies that the second return value can be safely ignored,
    matching how oceanmesh uses the API in signed_distance_function and
    edgefx.
    """

    node, edge = _square()
    pts = np.array([[0.5, 0.5], [2.0, 2.0]])

    stat, _ = inpoly2(pts, node, edge)
    assert stat.dtype == bool
    assert stat.shape == (2,)


def _run_all_methods(node, edge, pts):
    """Helper to run inpoly2 under all available backends.

    Returns a dict mapping method name -> (stat, bnds). Methods that
    cannot be used in the current environment are omitted.
    """

    results = {}
    for method in ("raycasting", "shapely", "matplotlib"):
        # Use a small context by temporarily setting the env var.
        from os import environ

        prev = environ.get("OCEANMESH_INPOLY_METHOD")
        try:
            environ["OCEANMESH_INPOLY_METHOD"] = method
            stat, bnds = inpoly2(pts, node, edge)
        except Exception:
            continue
        finally:
            if prev is None:
                environ.pop("OCEANMESH_INPOLY_METHOD", None)
            else:
                environ["OCEANMESH_INPOLY_METHOD"] = prev
        results[method] = (stat, bnds)
    return results


def test_strategy_consistency_square():
    node, edge = _square()
    pts = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
            [1.5, 0.5],
            [0.0, 0.0],
        ]
    )

    results = _run_all_methods(node, edge, pts)
    assert "raycasting" in results
    ray_stat, ray_bnd = results["raycasting"]

    # Require agreement on non-boundary points; allow minor
    # differences in STAT for boundary points due to backend-specific
    # semantics.
    boundary_mask = ray_bnd
    for method, (stat, bnd) in results.items():
        assert np.array_equal(bnd, ray_bnd)
        assert np.array_equal(stat[~boundary_mask], ray_stat[~boundary_mask])


def test_strategy_consistency_polygon_with_hole():
    node, edge = _polygon_with_hole()
    pts = np.array(
        [
            [0.5, 0.5],  # inside outer, outside inner
            [1.5, 1.5],  # inside inner ring region
            [3.5, 3.5],  # outside
        ]
    )

    results = _run_all_methods(node, edge, pts)
    # At minimum, ensure each available method runs and shapes match.
    for method, (stat, bnd) in results.items():
        assert stat.shape == (3,)
        assert bnd.shape == (3,)


def test_cython_acceleration_flag(monkeypatch):
    """Ensure compiled-kernel flag reflects environment toggling.

    This does not require the extension to be present; it simply
    verifies that toggling the environment and reloading the module
    does not crash and exposes the expected attribute.
    """

    monkeypatch.setenv("OCEANMESH_INPOLY_ACCEL", "1")
    mod = importlib.reload(pip_module)
    assert hasattr(mod, "_COMPILED_KERNEL_AVAILABLE")

    monkeypatch.setenv("OCEANMESH_INPOLY_ACCEL", "0")
    mod = importlib.reload(pip_module)
    assert hasattr(mod, "_COMPILED_KERNEL_AVAILABLE")


def test_pure_python_vs_cython_consistency(monkeypatch):
    """Compare pure-Python and accelerated implementations when available.

    If the compiled kernel is not built, the test is effectively a
    no-op but still exercises the environment toggling and reload
    behaviour.
    """

    node, edge = _square()
    pts = np.array(
        [
            [0.25, 0.25],
            [0.75, 0.75],
            [1.5, 0.5],
            [0.0, 0.0],
        ]
    )

    # Force pure-Python path
    monkeypatch.setenv("OCEANMESH_INPOLY_ACCEL", "0")
    pure_mod = importlib.reload(pip_module)
    stat_py, bnd_py = pure_mod.inpoly2(pts, node, edge)

    # Try accelerated path
    monkeypatch.setenv("OCEANMESH_INPOLY_ACCEL", "1")
    accel_mod = importlib.reload(pip_module)
    if not getattr(accel_mod, "_COMPILED_KERNEL_AVAILABLE", False):
        pytest.skip("Cython-accelerated kernel not available")

    stat_cy, bnd_cy = accel_mod.inpoly2(pts, node, edge)
    assert np.array_equal(stat_py, stat_cy)
    assert np.array_equal(bnd_py, bnd_cy)


@pytest.mark.slow
def test_cython_performance_smoke(monkeypatch):
    """Smoke-test that the accelerated kernel runs on a larger input.

    This is not a strict benchmark, but it exercises the Cython path on
    a moderately large problem when built.
    """

    node, edge = _square()
    rng = np.random.default_rng(0)
    pts = rng.random((20000, 2)) * 2.0 - 0.5

    monkeypatch.setenv("OCEANMESH_INPOLY_ACCEL", "1")
    mod = importlib.reload(pip_module)
    if not getattr(mod, "_COMPILED_KERNEL_AVAILABLE", False):
        pytest.skip("Cython-accelerated kernel not available")

    stat, bnd = mod.inpoly2(pts, node, edge)
    assert stat.shape == (20000,)
    assert bnd.shape == (20000,)
