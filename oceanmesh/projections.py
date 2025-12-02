"""Projection utilities for global stereographic meshing.

This module provides a small wrapper around cartopy's stereographic
projection for use in global meshes. The primary entry point is the
:class:`StereoProjection` class, which exposes methods for converting
between geographic (longitude/latitude) coordinates and a polar
stereographic Cartesian system.

Cartopy is treated as an optional dependency. The :data:`CARTOPY_AVAILABLE`
flag can be used to check availability at runtime, and the
:func:`check_cartopy_available` helper raises a clear, informative error
when global stereographic meshing is requested without cartopy being
installed.

The design of this module and the choice of defaults follow the
"global stereographic meshing" discussion from PR #87, with a
north-polar stereographic projection centred on 0° longitude and
true scale at the pole.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from pyproj import CRS, Transformer

try:  # pragma: no cover - exercised indirectly in integration tests
    import cartopy.crs as ccrs

    CARTOPY_AVAILABLE: bool = True
except Exception:  # pragma: no cover - when cartopy is not installed
    ccrs = None  # type: ignore[assignment]
    CARTOPY_AVAILABLE = False


class StereoProjection:
    r"""North-polar stereographic projection helper.

    This class provides forward (lon/lat -> x/y) and inverse (x/y -> lon/lat)
    transforms for a north-polar stereographic CRS, along with a helper for
    computing the local scale factor :math:`k(\phi)` using a configurable
    reference scale factor :math:`k_0`.

    Parameters
    ----------
    scale_factor:
        The reference scale factor :math:`k_0` used in the analytic
        expression for the stereographic scale distortion. For a standard
        north-polar stereographic projection this is typically 1.0.
    """

    def __init__(self, scale_factor: float = 1.0) -> None:
        if not CARTOPY_AVAILABLE:  # Defensive check; public callers
            check_cartopy_available()  # will usually call this first.

        # Geographic CRS for lon/lat coordinates.
        self._pc = CRS.from_epsg(4326)
        # North-polar stereographic CRS used for global meshes. Use a
        # definition aligned with cartopy's NorthPolarStereo defaults.
        self._stereo = CRS.from_proj4(
            "+proj=stere +lat_0=90 +lat_ts=90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
        )
        # pyproj transformer for forward/inverse operations.
        self._fwd = Transformer.from_crs(self._pc, self._stereo, always_xy=True)
        self._inv = Transformer.from_crs(self._stereo, self._pc, always_xy=True)
        self.scale_factor = float(scale_factor)

    @property
    def crs_geographic(self):
        """Return the underlying geographic CRS (lon/lat)."""

        return self._pc

    @property
    def crs_stereo(self):
        """Return the underlying stereographic CRS."""

        return self._stereo

    def to_stereo(
        self, lon: np.ndarray, lat: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project longitude/latitude to stereographic x/y.

        Parameters
        ----------
        lon, lat:
            Arrays of longitudes and latitudes in degrees.

        Returns
        -------
        x, y:
            Arrays of stereographic coordinates in the projection's
            native units (typically metres).
        """

        x, y = self._fwd.transform(lon, lat)
        return x, y

    def to_lat_lon(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Inverse transform from stereographic x/y to lon/lat.

        Parameters
        ----------
        x, y:
            Arrays of stereographic coordinates in the projection's
            native units (typically metres).

        Returns
        -------
        lon, lat:
            Arrays of longitudes and latitudes in degrees.
        """

        lon, lat = self._inv.transform(x, y)
        return lon, lat

    def get_scale_factor(self, lat: np.ndarray) -> np.ndarray:
        r"""Return the stereographic scale factor at the given latitude.

        The scale factor :math:`k(\phi)` for a polar stereographic
        projection is given by

        .. math::

            k(\phi) = \frac{2 k_0}{1 + \sin \phi},

        where :math:`k_0` is a reference scale factor. Here ``lat`` is
        given in degrees and internally converted to radians.

        Parameters
        ----------
        lat:
            Latitude or array of latitudes in degrees.

        Returns
        -------
        k:
            The corresponding scale factor(s).
        """

        lat_rad = np.deg2rad(lat)
        return 2.0 * self.scale_factor / (1.0 + np.sin(lat_rad))


def check_cartopy_available() -> None:
    """Raise an informative error if cartopy is not installed.

    This helper is intended to be called from high-level entry points
    such as :class:`oceanmesh.geodata.Shoreline` when ``stereo=True``
    is requested. It ensures that users receive a clear error message
    explaining how to enable the optional global meshing dependencies
    instead of encountering opaque import or runtime failures later in
    the workflow.
    """

    if not CARTOPY_AVAILABLE:
        msg = (
            "Global stereographic meshing (stereo=True) requires cartopy. "
            "Install it with: pip install oceanmesh[global] or pip install cartopy"
        )
        raise ImportError(msg)


__all__ = ["StereoProjection", "check_cartopy_available", "CARTOPY_AVAILABLE"]
