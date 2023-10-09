import numpy as np
from pyproj import CRS, Transformer

__all__ = ["Region", "warp_coordinates"]


def warp_coordinates(points, src_crs, dst_crs):
    src_crs = CRS.from_epsg(src_crs)
    dst_crs = CRS.from_epsg(dst_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    points = transformer.transform(points[:, 0], points[:, 1])
    return np.asarray(points).T


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
    kx, ky, kz = to_3d(x, y, R)

    # to 2D in stereo
    #    u = 2*R*kx/(R+kz)
    #    v = 2*R*ky/(R+kz)
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
