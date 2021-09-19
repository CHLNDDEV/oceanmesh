from pyproj import CRS, Transformer


class BoundingBox:
    def __init__(self, extent, crs):
        self.bbox = extent
        self._crs = CRS.from_user_input(crs)

    @property
    def crs(self):
        return self._crs

    @property
    def bbox(self):
        return self.__bbox

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
        """Transform to dst_crs"""
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
                self.bbox = transformer.transform(self.bbox)

            self._crs = dst_crs
