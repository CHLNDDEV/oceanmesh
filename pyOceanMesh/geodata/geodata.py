import os, errno

import numpy
import numpy.ma as ma
import matplotlib.path as mpltPath

from netCDF4 import Dataset
import shapefile


class Geodata:
    """
    Parent geographical data class that handles geographical data describing
    coastlines or other features in the form of a shapefile and
    topobathy in the form of a DEM.
    """

    def __init__(self, bbox):
        self.bbox = bbox

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


def classify_shoreline(bbox, polys):
    """Classify vertices from polys as either inner, outer, or mainland.
       The vertices of the shoreline polygon(s) are classified into three types: mainland, inner, or outer.
        (a) The mainland category contains vertices that are not totally enclosed inside the bbox.
        (b) The inner (i.e., islands) category contains *polyons* totally enclosed inside the bbox.
        (c) The outer category is the union of the mainland, inner, and bbox polygons.
    """
    # path of bounding box
    def __create_boubox(bbox):
        xmin, xmax, ymin, ymax = bbox
        return [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]

    inner = numpy.empty(shape=(0, 2))
    mainland = numpy.empty(shape=(0, 2))
    outer = numpy.empty(shape=(0, 2))

    path = mpltPath.Path(__create_boubox(bbox))
    for poly in polys:
        inside = path.contains_points(poly[:-1, :])
        if all(inside):
            # an island/inner polygon
            inner = numpy.append(inner, poly, axis=0)
        elif any(inside):
            # a mainland polyline/polygon (may be non-closed)
            mainland = numpy.append(mainland, poly, axis=0)
    return inner, mainland, outer


class Shoreline(Geodata):
    """Repr. of shoreline
       The shoreline is represented as a list of numpy array of winding points with
       mask values representing breaks.
    """

    def __init__(self, shp, bbox):
        super().__init__(bbox)

        self.shp = shp
        self.inner = []
        self.outer = []
        self.mainland = []

        @property
        def shp(self):
            return self.__shp

        @shp.setter
        def shp(self, fname):
            if not os.path.isfile(fname):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)
            self.__shp = fname

        polys = []  # tmp storage for polygons and polylines

        def __isOverlapping(bbox1, bbox2):
            x1min, x1max, y1min, y1max = bbox1
            x2min, x2max, y2min, y2max = bbox2
            return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

        print("Reading in shapefile: " + self.shp)
        s = shapefile.Reader(self.shp)
        for shape in s.shapes():
            # only read in shapes that intersect with bbox
            if __isOverlapping(bbox, shape.bbox):
                poly = numpy.asarray(shape.points + [(999.0, 999.0)])
                poly = ma.masked_where(poly == 999.0, poly)
                polys.append(poly)
        if len(polys) == 0:
            raise ValueError("Shoreline does not intersect bbox")

        self.inner, self.mainland, self.outer = classify_shoreline(self.bbox, polys)

    def plot(self, hold_on=False):
        """plot the content of the shp field"""
        import matplotlib.pyplot as plt

        tmp = numpy.concatenate(self.polys, axis=0)
        tmp = ma.masked_where(tmp == 999.0, tmp)

        fig, ax = plt.subplots()
        ax.plot(tmp[:, 0], tmp[:, 1], "k-")
        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, facecolor="red", alpha=0.1
        )

        border = 0.10 * (xmax - xmin)
        plt.xlim(xmin - border, xmax + border)
        plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        plt.show()


class DEM(Geodata):
    """Repr. of digitial elevation model"""

    def __init__(self, dem, bbox):
        super().__init__(bbox)

        self.dem = dem

        @property
        def dem(self):
            return self.__dem

        @dem.setter
        def dem(self, fname):
            if not os.path.isfile(fname):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), fname)
            self.__dem = fname

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
            print("Unable to open file.")
