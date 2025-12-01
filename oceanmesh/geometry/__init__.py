"""Geometry utilities for oceanmesh.

This subpackage currently provides a GPL-compatible 2D point-in-polygon
implementation via :func:`inpoly2`, intended as a drop-in replacement
for the vendored `inpoly-python` library.
"""

from .point_in_polygon import inpoly2

__all__ = ["inpoly2"]
