class Geodata:
    """
    GEODATA: Geographical data class
    Handles geographical data describing coastlines or other features in
    the form of a shapefile and topobathy in the form of a DEM
    """

    def __init__(self, bbox=None):
        self.bbox = bbox

    @property
    def bbox(self):
        return self.__bbox

    @bbox.setter
    def bbox(self, value):
        if value is None:
            self.__bbox = value
        else:
            assert len(value) <= 4, "bbox has wrong number of values."
            self.__bbox = value
