import numpy as np
from pyproj import CRS, Transformer

__all__ = ["Region", "warp_coordinates"]


def warp_coordinates(points, src_crs, dst_crs):
    src_crs = CRS.from_epsg(src_crs)
    dst_crs = CRS.from_epsg(dst_crs)
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    points = transformer.transform(points[:, 0], points[:, 1])
    return np.asarray(points).T


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
