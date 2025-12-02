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


def is_global_bbox(bbox):
    """Heuristic to determine if a bbox is 'global-like'.

    Returns True when bbox covers most of the world extents.
    Expected bbox format: (xmin, xmax, ymin, ymax) in degrees.
    """
    xmin, xmax, ymin, ymax = bbox
    lon_span = xmax - xmin
    lat_span = ymax - ymin
    lon_global = lon_span >= 300 or (xmin <= -180 and xmax >= 180)
    lat_global = lat_span >= 150 or (ymin <= -89 and ymax >= 89)
    return bool(lon_global and lat_global)


def bbox_contains(outer, inner):
    """Return True if outer bbox fully contains inner bbox.

    Both bboxes are (xmin, xmax, ymin, ymax). If the outer bbox is detected
    as global using is_global_bbox, containment is assumed True. If the inner
    bbox is global and outer is not, containment is False.
    """
    if is_global_bbox(outer):
        return True
    if is_global_bbox(inner) and not is_global_bbox(outer):
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
        r = CRS.from_user_input(regional_crs)
    except Exception as e:
        return (
            False,
            f"Failed to parse CRS. global={get_crs_string(global_crs)}, regional={get_crs_string(regional_crs)}; error={e}",
        )

    # Global must be EPSG:4326 (WGS84 geographic)
    try:
        g_auth = g.to_epsg()
    except Exception:
        g_auth = None
    if not g.is_geographic or g_auth != 4326:
        return False, f"Global domain must be EPSG:4326, found {get_crs_string(g)}"

    # Regional can be geographic or projected; both acceptable
    if r.is_geographic:
        return True, "Compatible CRS: global EPSG:4326 with regional geographic CRS"
    if r.is_projected:
        return True, "Compatible CRS: global EPSG:4326 with regional projected CRS"

    # Fallback neutral
    return True, "CRS types appear compatible"


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
    scale_factor:
        Reference stereographic scale factor :math:`k_0` used in the
        distortion formula. Projections are cached per distinct k0
        value when cartopy/pyproj are available.
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
