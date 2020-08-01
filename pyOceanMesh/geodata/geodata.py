class Geodata:
    """
    Geographical data class
    Handles geographical data describing coastlines or other features in
    the form of a shapefile and topobathy in the form of a DEM
    """

    def __init__(self, shp=None, dem=None, bbox=None):
        self.bbox = bbox
        self.shp = shp
        self.dem = dem

    @property
    def bbox(self):
        return self.__bbox

    @bbox.setter
    def bbox(self, value):
        if value is None:
            self.__bbox = value
        else:
            if len(value) < 4:
                raise ValueError("bbox has wrong number of values.")
            self.__bbox = value


class Shoreline(Geodata):
    """Repr. of shoreline"""

    def __init__(self, shp, bbox):
        super().__init__(shp=shp, bbox=bbox)


class DEM(Geodata):
    """Repr. of digitial elevation model"""

    def __init__(self, dem, bbox):
        super().__init__(dem=dem, bbox=bbox)
        self.topobathy = None
