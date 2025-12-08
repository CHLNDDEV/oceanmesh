r"""Coordinate-system utilities and stereographic helpers for OceanMesh.

This module centralises logic for working with geographic and
projected coordinate reference systems (CRS), including:

* Lightweight wrappers around :mod:`pyproj` for CRS parsing and
    coordinate warping.
* Convenience helpers for detecting "global-like" bounding boxes and
    performing CRS-aware containment checks used in multiscale mesh
    validation.
* Optional integration with the cartopy-backed
    :class:`~oceanmesh.projections.StereoProjection` class for
    north-polar stereographic projections.

Stereographic projections for global meshes
------------------------------------------

For global ocean meshes, OceanMesh uses a north-polar stereographic
projection: geographic coordinates (lon, lat) are projected onto a
plane (u, v) tangent at the north pole. When cartopy is available,
this is implemented via its ``NorthPolarStereo`` CRS combined with a
``PlateCarree`` (lon/lat) CRS. The forward and inverse transforms are
exposed through :func:`to_stereo` and :func:`to_lat_lon`.

The local scale factor :math:`k(\phi)` of a polar stereographic
projection, which describes how distances on the plane relate to
distances on the sphere at latitude :math:`\phi`, is given by

.. math::

    k(\phi) = \frac{2 k_0}{1 + \sin \phi},

where :math:`k_0` is a configurable reference scale factor (typically
1.0). A commonly used value in practical ocean and ice modelling is
``k0 = 0.994``, mirroring conventions in EPSG:3413 (Arctic) and
EPSG:3031 (Antarctic) to reduce distortion near standard parallels.

When cartopy is not installed, OceanMesh falls back to legacy analytic
formulas based on a unit sphere. These formulas are consistent with
the mapping used in the original Fortran/C implementations and remain
useful for environments where cartopy is difficult to install,
albeit with less CRS metadata.

Background and references
-------------------------

The configurable scale-factor support (:math:`k_0`) and cartopy-based
projection backend were introduced in PR #87:

        https://github.com/CHLNDDEV/oceanmesh/pull/87

A closely related implementation can be found in the seamsh example:

        https://git.immc.ucl.ac.be/jlambrechts/seamsh/-/blob/ca380b59fe4d2ea57ffbf08a7b0c70bdf7df1afb/examples/6-stereographics.py#L55

For theoretical background and additional formulas, see

        Snyder, J.P. (1987). *Map Projections – A Working Manual*.
        U.S. Geological Survey Professional Paper 1395.
"""

import numpy as np
from pyproj import CRS, Transformer

from .projections import CARTOPY_AVAILABLE, StereoProjection

__all__ = [
    "Region",
    "warp_coordinates",
    "to_stereo",
    "to_lat_lon",
    "stereo_to_3d",
    "to_3d",
]


# ---- Helpers for CRS and bbox validation in multiscale meshes ----
def get_crs_string(crs):
    """Return a compact string representation of a CRS-like object.

    Parameters
    ----------
    crs : pyproj.CRS | str | None
    """
    if crs is None:
        return "None"
    try:
        _crs = CRS.from_user_input(crs)
        return _crs.to_string()
    except Exception:
        return str(crs)


def is_global_bbox(bbox, crs=None):
    """Heuristic to determine if a bbox is 'global-like'.

    Parameters
    ----------
    bbox : tuple
        (xmin, xmax, ymin, ymax) in degrees or projection units.
    crs : optional
        CRS of the bbox. When provided and projected, looser thresholds
        are used to decide whether the bbox is effectively global.
    """

    xmin, xmax, ymin, ymax = bbox
    lon_span = xmax - xmin
    lat_span = ymax - ymin

    if crs is not None:
        try:
            _crs = CRS.from_user_input(crs)
        except Exception:
            _crs = None
        if _crs is not None and _crs.is_projected:
            # For projected CRS we rely purely on span heuristics in
            # the projected units. Large extents compared to Earth
            # radius suggest global coverage.
            span_x = abs(lon_span)
            span_y = abs(lat_span)
            threshold = 5_000_000.0  # ~5000 km
            return bool(span_x >= threshold and span_y >= threshold)

    lon_global = lon_span >= 300 or (xmin <= -180 and xmax >= 180)
    lat_global = lat_span >= 150 or (ymin <= -89 and ymax >= 89)
    return bool(lon_global and lat_global)


def bbox_contains(outer, inner, outer_crs=None, inner_crs=None):
    """Return True if outer bbox fully contains inner bbox.

    Both bboxes are (xmin, xmax, ymin, ymax). If CRS information is
    provided and differs, the inner bbox is transformed into the outer
    CRS before checking containment. For global-like outer bboxes,
    containment is assumed True. If the inner bbox is global and outer
    is not, containment is False.
    """

    if outer_crs is not None and inner_crs is not None:
        try:
            o = CRS.from_user_input(outer_crs)
            i = CRS.from_user_input(inner_crs)
        except Exception:
            o = i = None
        if o is not None and i is not None and not o.equals(i):
            transformer = Transformer.from_crs(i, o, always_xy=True)
            ixmin, ixmax, iymin, iymax = inner
            xs = [ixmin, ixmax, ixmax, ixmin]
            ys = [iymin, iymin, iymax, iymax]
            tx, ty = transformer.transform(xs, ys)
            inner = (
                float(min(tx)),
                float(max(tx)),
                float(min(ty)),
                float(max(ty)),
            )

    if is_global_bbox(outer, outer_crs):
        return True
    if is_global_bbox(inner, inner_crs) and not is_global_bbox(outer, outer_crs):
        return False

    oxmin, oxmax, oymin, oymax = outer
    ixmin, ixmax, iymin, iymax = inner
    return (
        (oxmin <= ixmin) and (oxmax >= ixmax) and (oymin <= iymin) and (oymax >= iymax)
    )


def validate_crs_compatible(global_crs, regional_crs):
    """Validate that a global and regional CRS are compatible for mixing.

    Rules:
      - Global domain must be WGS84 geographic (EPSG:4326)
      - Regional may be EPSG:4326 or a projected CRS
      - If either CRS is missing, report soft-pass with guidance

    Returns
    -------
    (ok: bool, msg: str)
    """
    if global_crs is None or regional_crs is None:
        return True, "CRS metadata missing; skipping strict compatibility check"

    try:
        g = CRS.from_user_input(global_crs)
        CRS.from_user_input(regional_crs)
    except Exception as e:
        return (
            False,
            f"Failed to parse CRS. global={get_crs_string(global_crs)}, regional={get_crs_string(regional_crs)}; error={e}",
        )

    # Global should be geographic (preferably EPSG:4326) or a
    # global-suitable projected CRS.
    try:
        g_auth = g.to_epsg()
    except Exception:
        g_auth = None

    if g.is_geographic:
        if g_auth != 4326:
            return (
                True,
                f"Global domain uses geographic CRS {get_crs_string(g)} (not EPSG:4326). "
                "This may work but EPSG:4326 is recommended for global meshing.",
            )
        return True, "Compatible CRS: global EPSG:4326 with regional CRS"

    if g.is_projected:
        proj_desc = g.to_proj4()
        if isinstance(proj_desc, str) and ("stere" in proj_desc.lower()):
            return (
                True,
                f"Global domain uses stereographic projection {get_crs_string(g)}, which is suitable for global meshing.",
            )
        return (
            True,
            f"Global domain uses projected CRS {get_crs_string(g)}. Ensure this projection is suitable for global-scale meshing.",
        )

    return (
        False,
        f"Global domain CRS {get_crs_string(g)} is neither geographic nor projected; cannot validate compatibility.",
    )


def is_crs_suitable_for_global(crs):
    """Return (is_suitable, message) for a CRS used as global.

    This helper prefers EPSG:4326, accepts other geographic CRS with a
    warning, and treats stereographic projections as suitable for
    global stereographic meshing.
    """

    if crs is None:
        return False, "No CRS supplied for global domain."

    try:
        c = CRS.from_user_input(crs)
    except Exception as e:
        return False, f"Failed to parse CRS {crs!r}: {e}"

    try:
        auth = c.to_epsg()
    except Exception:
        auth = None

    if c.is_geographic and auth == 4326:
        return True, "Ideal global CRS: EPSG:4326 (WGS84 geographic)."

    if c.is_geographic:
        return (
            True,
            f"Global domain uses geographic CRS {get_crs_string(c)} (not EPSG:4326); acceptable but EPSG:4326 is recommended.",
        )

    if c.is_projected:
        proj_desc = c.to_proj4()
        if isinstance(proj_desc, str) and ("stere" in proj_desc.lower()):
            return (
                True,
                f"Global domain uses stereographic projection {get_crs_string(c)}, suitable for global stereographic meshing.",
            )
        return (
            True,
            f"Global domain uses projected CRS {get_crs_string(c)}; ensure it is appropriate for global-scale meshing.",
        )

    return (
        False,
        f"CRS {get_crs_string(c)} is neither geographic nor projected; not suitable for global meshing.",
    )


def warp_coordinates(points, src_crs, dst_crs):
    src_crs = CRS.from_epsg(src_crs)
    dst_crs = CRS.from_epsg(dst_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    points = transformer.transform(points[:, 0], points[:, 1])
    return np.asarray(points).T


_STEREO_PROJ_CACHE = {}


def _get_stereo_projection(scale_factor=1.0):
    """Return a cached :class:`StereoProjection` for a given k0.

    Parameters
    ----------
    scale_factor : float, optional
        Reference stereographic scale factor :math:`k_0` used in the
        distortion formula ``k(phi) = 2*k0 / (1 + sin(phi))``.
        Projections are cached per distinct :math:`k_0` value when
        cartopy/pyproj are available.

    Notes
    -----
    Caching avoids repeatedly constructing cartopy CRS/Transformer
    objects during mesh generation and gradient enforcement. Different
    :math:`k_0` values lead to different local scale factors near the
    poles and thus different effective mesh densities.
    """

    k0_key = float(scale_factor)
    if k0_key not in _STEREO_PROJ_CACHE and CARTOPY_AVAILABLE:
        _STEREO_PROJ_CACHE[k0_key] = StereoProjection(scale_factor=k0_key)
    return _STEREO_PROJ_CACHE.get(k0_key, None)


def stereo_to_3d(u, v, R=1):
    # to 3D
    #    c=4*R**2/(u**2+v**2+4*R**2)
    #    x=c*u
    #    y=c*v
    #    z=2*c*R-R

    rp2 = u**2 + v**2
    x = -2 * R * u / (1 + rp2)
    y = -2 * R * v / (1 + rp2)
    z = R * (1 - rp2) / (1 + rp2)

    return x, y, z


def to_lat_lon(x, y, z=None, R=1):
    """Convert stereographic or 3D Cartesian coordinates to lon/lat.

    When cartopy is available and ``z is None`` with the default
    radius ``R == 1``, this uses the cartopy-based
    :class:`StereoProjection` inverse transform. Otherwise it falls
    back to the legacy analytic formulas for backward compatibility.
    """

    if z is None and R == 1 and CARTOPY_AVAILABLE:
        proj = _get_stereo_projection()
        if proj is not None:
            lon, lat = proj.to_lat_lon(np.asarray(x), np.asarray(y))
            return lon, lat

    if z is None:
        x, y, z = stereo_to_3d(x, y, R=R)

    # to lat/lon
    rad = x**2 + y**2 + z**2
    rad = np.sqrt(rad)

    rad[rad == 0] = rad.max()

    rlat = np.arcsin(z / rad)
    rlon = np.arctan2(y, x)

    rlat = rlat * 180 / np.pi
    rlon = rlon * 180 / np.pi

    return rlon, rlat


def to_3d(x, y, R=1):
    lon = np.array(x)
    lat = np.array(y)
    # to 3D
    kx = np.cos(lat / 180 * np.pi) * np.cos(lon / 180 * np.pi) * R
    ky = np.cos(lat / 180 * np.pi) * np.sin(lon / 180 * np.pi) * R
    kz = np.sin(lat / 180 * np.pi) * R

    return kx, ky, kz


def to_stereo(x, y, R=1):
    """Project lon/lat to stereographic coordinates.

    When cartopy is available and the default radius ``R == 1`` is
    used, this delegates to the cartopy-based :class:`StereoProjection`
    helper. Otherwise, it falls back to the legacy analytic formulas
    based on a unit sphere, preserving historical behaviour.
    """

    if R == 1 and CARTOPY_AVAILABLE:
        proj = _get_stereo_projection()
        if proj is not None:
            u, v = proj.to_stereo(np.asarray(x), np.asarray(y))
            return u, v

    kx, ky, kz = to_3d(x, y, R)

    # to 2D in stereo (legacy formulation)
    u = -kx / (R + kz)
    v = -ky / (R + kz)

    return u, v


class Region:
    def __init__(self, extent, crs):
        self.bbox = extent
        self._crs = CRS.from_user_input(crs)

    @property
    def crs(self):
        return self._crs

    @property
    def bbox(self):
        return self.__bbox

    @property
    def total_bounds(self):
        if isinstance(self.bbox, tuple):
            return self.bbox
        else:
            return (
                self.bbox[:, 0].min(),
                self.bbox[:, 0].max(),
                self.bbox[:, 1].min(),
                self.bbox[:, 1].max(),
            )

    @bbox.setter
    def bbox(self, value):
        if isinstance(value, tuple):
            if len(value) < 4:
                raise ValueError("bbox has wrong number of values.")
            if value[1] < value[0]:
                raise ValueError("bbox has wrong values.")
            if value[3] < value[2]:
                raise ValueError("bbox has wrong values.")
        # otherwise polygon
        self.__bbox = value

    def transform_to(self, dst_crs):
        """Transform extents ``bbox`` to dst_crs"""
        dst_crs = CRS.from_user_input(dst_crs)
        if not self._crs.equals(dst_crs):
            transformer = Transformer.from_crs(self.crs, dst_crs, always_xy=True)
            if isinstance(self.bbox, tuple):
                xmin, xmax, ymin, ymax = self.bbox
                (xmin, xmax), (ymin, ymax) = transformer.transform(
                    (xmin, xmax), (ymin, ymax)
                )
                self.bbox = (xmin, xmax, ymin, ymax)
            else:
                # for polygon case
                self.bbox = np.asarray(
                    transformer.transform(self.bbox[:, 0], self.bbox[:, 1])
                ).T

            self._crs = dst_crs
        return self
