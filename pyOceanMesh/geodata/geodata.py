import os, errno

import numpy
import numpy.ma as ma
import matplotlib.path as mpltPath

from netCDF4 import Dataset
import shapefile


def create_boubox(bbox):
    xmin, xmax, ymin, ymax = bbox
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]


def densify(poly, maxdiff):
    """ Fills in any gaps in latitude (lat) or longitude (lon) data vectors
        that are greater than a defined tolerance maxdiff (degrees) apart in either dimension.

        latout and lonout are the new latitude and longitude data vectors, in which any gaps
        larger than maxdiff in the original vectors have been filled with additional points.
    """
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = numpy.abs(lat[1:] - lat[:-1])
    dlon = numpy.abs(lon[1:] - lon[:-1])
    nin = numpy.ceil(numpy.maximum(dlat, dlon) / maxdiff) - 1
    sumnin = numpy.sum(nin)
    if sumnin == 0:
        print("No insertion is needed")
        return lon, lat
    nout = sumnin + nx
    latout = ma.array(numpy.arange(nout), mask=1)
    lonout = ma.array(numpy.arange(nout), mask=1)
    n = 0
    for i in range(nx - 1):
        ni = nin[i]
        if ni == 0 or ma.is_masked(ni):
            latout[n] = lat[i]
            lonout[n] = lon[i]
            nstep = 1
        else:
            ni = int(ni)
            ilat = numpy.linspace(lat[i], lat[i + 1], ni + 2)
            ilon = numpy.linspace(lon[i], lon[i + 1], ni + 2)
            latout[n : n + ni] = ilat[1 : ni + 1]
            lonout[n : n + ni] = ilon[1 : ni + 1]
            nstep = ni + 1
        n += nstep

    latout[-1] = lat[-1]
    lonout[-1] = lon[-1]
    return numpy.hstack((lonout, latout))


def polyArea(x, y):
    return 0.5 * numpy.abs(
        numpy.dot(x, numpy.roll(y, 1)) - numpy.dot(y, numpy.roll(x, 1))
    )


def isOverlapping(bbox1, bbox2):
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def classifyShoreline(bbox, polys, h0):
    """Classify vertices from polys as either inner or mainland.
        (a) The mainland category contains vertices that are not totally enclosed inside the bbox.
        (b) The inner (i.e., islands) category contains *polyons* totally enclosed inside the bbox.
       NB: Removes islands with area smaller than 4*h0**2
    """
    print("Partitioning shoreline...")

    boubox = create_boubox(bbox)

    inner = numpy.empty(shape=(0, 2))
    mainland = numpy.empty(shape=(0, 2))

    path = mpltPath.Path(boubox)
    for poly in polys:
        inside = path.contains_points(poly[:-1, :])
        if all(inside):
            # compute area of polygon and don't save if too small
            area = polyArea(poly[:-1, 0], poly[:-1, 1])
            if area < 4 * h0 ** 2:
                print("Island skipped...area too small for given h0")
                continue
            inner = numpy.append(inner, poly, axis=0)
            inner = ma.masked_where(inner == 999.0, inner)
        elif any(inside):
            # a mainland polyline/polygon (may be non-closed)
            mainland = numpy.append(mainland, poly, axis=0)
            mainland = ma.masked_where(mainland == 999.0, mainland)

    return inner, mainland


class Geodata:
    """
    Parent geographical data class that handles geographical data describing
    coastlines or other features in the form of a shapefile and
    topobathy in the form of a DEM.
    """

    def __init__(self, bbox, h0):
        self.bbox = bbox
        self.h0 = h0

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
            if value[1] < value[0]:
                raise ValueError("bbox has wrong values.")
            if value[3] < value[2]:
                raise ValueError("bbox has wrong values.")
            self.__bbox = value

    @property
    def h0(self):
        return self.__h0

    @h0.setter
    def h0(self, value):
        if value <= 0:
            raise ValueError("h0 must be > 0")
        value /= 111e3  # convert to wgs84 degrees
        self.__h0 = value


class Shoreline(Geodata):
    """Repr. of shoreline
       The shoreline is represented as a list of numpy array
       of winding points with mask values representing breaks.
    """

    def __init__(self, shp, bbox, h0):
        super().__init__(bbox, h0)

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

        print("Reading in shapefile... " + self.shp)
        s = shapefile.Reader(self.shp)
        for shape in s.shapes():
            # only read in shapes that intersect with bbox
            if isOverlapping(bbox, shape.bbox):
                poly = numpy.asarray(shape.points + [(999.0, 999.0)])
                poly = ma.masked_where(poly == 999.0, poly)
                polys.append(poly)

        if len(polys) == 0:
            raise ValueError("Shoreline does not intersect bbox")

        inner, mainland = classifyShoreline(self.bbox, polys, self.h0)

        # densification of point spacing
        self.inner = densify(inner, self.h0)
        self.mainland = densify(mainland, self.h0)

        # apply smoother (if active)

    def plot(self, hold_on=False):
        """plot the content of the shp field"""
        import matplotlib.pyplot as plt

        flg1, flg2 = False, False

        fig, ax = plt.subplots()
        if len(self.mainland) != 0:
            (line1,) = ax.plot(self.mainland[:, 0], self.mainland[:, 1], "g-")
            flg1 = True
        if len(self.inner) != 0:
            (line2,) = ax.plot(self.inner[:, 0], self.inner[:, 1], "r-")
            flg2 = True

        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin, fill=None, hatch="///", alpha=0.2,
        )

        border = 0.10 * (xmax - xmin)
        plt.xlim(xmin - border, xmax + border)
        plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        if flg1 and flg2:
            ax.legend((line1, line2), ("mainland", "inner"))
        elif flg1 and not flg2:
            ax.legend((line1), ("mainland"))
        elif flg2 and not flg1:
            ax.legend((line1), ("inner"))

        plt.show()


class DEM(Geodata):
    """Repr. of digitial elevation model"""

    def __init__(self, dem, bbox, h0):
        super().__init__(bbox, h0)

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
