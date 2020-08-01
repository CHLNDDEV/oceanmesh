from netCDF4 import Dataset


class Geodata:
    """
    Geographical data class that handles geographical data describing
    coastlines or other features in the form of a shapefile and
    topobathy in the form of a DEM.
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
        # well-known variable names
        wkv_x = ["x", "Longitude", "longitude", "lon"]
        wkv_y = ["y", "Latitude", "latitude", "lat"]
        wkv_z = ["Band1", "z"]
        try:
            with Dataset(dem, "r") as nc_fid:
                for x, y in zip(wkv_x, wkv_y):
                    for var in nc_fid.variables:
                        if var == x:
                            lon_name = var
                        if var == y:
                            lat_name = var
                for z in wkv_z:
                    for var in nc_fid.variables:
                        if var == z:
                            z_name = var
                self.lats = nc_fid.variables[lon_name][:]
                self.lons = nc_fid.variables[lat_name][:]
                self.topobathy = nc_fid.variables[z_name][:]
        except IOError:
            print("Unable to open file. Not found or read permissions incorrect")
