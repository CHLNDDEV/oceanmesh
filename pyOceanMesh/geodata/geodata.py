import os
import errno

import numpy
import matplotlib.path as mpltPath

import shapefile
from netCDF4 import Dataset

nan = numpy.nan

__all__ = ["Geodata", "Shoreline", "DEM"]


def _convert_to_array(lst):
    """Converts a list of Numpy arrays to a Numpy array"""
    return numpy.concatenate(lst, axis=0)


def _convert_to_list(arr):
    """Converts a nan-delimited Numpy array to a list of Numpy arrays"""
    a = numpy.insert(arr, 0, [[nan, nan]], axis=0)
    tmp = [a[s] for s in numpy.ma.clump_unmasked(numpy.ma.masked_invalid(a[:, 0]))]
    return [numpy.append(a, [[nan, nan]], axis=0) for a in tmp]


def _create_boubox(bbox):
    """Create a bounding box from domain extents `bbox`."""
    xmin, xmax, ymin, ymax = bbox
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
    ]


def _create_ranges(start, stop, N, endpoint=True):
    """Vectorized alternative to numpy.linspace"""
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * numpy.arange(N) + start[:, None]


def _densify(poly, maxdiff):
    """ Fills in any gaps in latitude or longitude arrays
        that are greater than a `maxdiff` (degrees) apart
    """
    print("Densifying segments...")
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = numpy.abs(lat[1:] - lat[:-1])
    dlon = numpy.abs(lon[1:] - lon[:-1])
    nin = numpy.ceil(numpy.maximum(dlat, dlon) / maxdiff) - 1
    sumnin = numpy.nansum(nin)
    if sumnin == 0:
        print("No densification is needed")
        return numpy.hstack((lon[:, None], lat[:, None]))
    nout = sumnin + nx

    lonout = numpy.full((int(nout)), nan, dtype=float)
    latout = numpy.full((int(nout)), nan, dtype=float)

    n = 0
    for i in range(nx - 1):
        ni = nin[i]
        if ni == 0 or numpy.isnan(ni):
            latout[n] = lat[i]
            lonout[n] = lon[i]
            nstep = 1
        else:
            ni = int(ni)
            icoords = _create_ranges(
                numpy.array([lat[i], lon[i]]),
                numpy.array([lat[i + 1], lon[i + 1]]),
                ni + 2,
            )
            latout[n : n + ni + 1] = icoords[0, : ni + 1]
            lonout[n : n + ni + 1] = icoords[1, : ni + 1]
            nstep = ni + 1
        n += nstep

    latout[-1] = lat[-1]
    lonout[-1] = lon[-1]
    return numpy.hstack((lonout[:, None], latout[:, None]))


def _polyArea(x, y):
    """Calculates area of a polygon"""
    return 0.5 * numpy.abs(
        numpy.dot(x, numpy.roll(y, 1)) - numpy.dot(y, numpy.roll(x, 1))
    )


def _isOverlapping(bbox1, bbox2):
    """Determines if two boxes intersect"""
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def _classifyShoreline(bbox, polys, h0, minimum_area_mult):
    """Classify segments in numpy.array `polys` as either `inner` or `mainland`.
        (1) The `mainland` category contains segments that are not totally enclosed inside the `bbox`.
        (2) The `inner` (i.e., islands) category contains segments totally enclosed inside the `bbox`.
            NB: Removes `inner` geometry with areas smaller than `minimum_area_mult`*`h0`**2
    """
    print("Classifying the shoreline segments...")

    boubox = _create_boubox(bbox)

    inner = numpy.empty(shape=(0, 2))
    inner[:] = nan
    mainland = numpy.empty(shape=(0, 2))
    mainland[:] = nan

    polys = _convert_to_list(polys)
    path = mpltPath.Path(boubox)
    for poly in polys:
        inside = path.contains_points(poly[:-2])
        if all(inside):
            area = _polyArea(poly[:-2, 0], poly[:-2, 1])
            if area < minimum_area_mult * h0 ** 2:
                continue
            inner = numpy.append(inner, poly, axis=0)
        elif any(inside):
            mainland = numpy.append(mainland, poly, axis=0)

    return inner, mainland


def chaikins_corner_cutting(coords, refinements=5):
    """http://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/Chaikins-Algorithm.pdf"""
    coords = numpy.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = numpy.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords


def _smoothShoreline(polys, N):
    """Smoothes the shoreline segment-by-segment using
       a `N` refinement Chaikins Corner cutting algorithm.
    """
    print("Applying a {} refinement Chaikin Corner cut to segments...".format(N))
    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        tmp = chaikins_corner_cutting(poly[:-1], refinements=N)
        tmp = numpy.append(tmp, [[nan, nan]], axis=0)
        out.append(tmp)
    return _convert_to_array(out)


def _nth_simplify(polys, bbox):
    """Collapse segments in `polys` outside of `bbox`"""
    print("Collapsing segments outside of the bbox...")
    boubox = _create_boubox(bbox)
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        j = 0
        inside = path.contains_points(poly[:-2, :])
        line = numpy.empty(shape=(0, 2))
        while j < len(poly[:-2]):
            if inside[j]:  # keep point (in domain)
                line = numpy.append(line, [poly[j, :]], axis=0)
            else:  # pt is outside of domain
                bd = min(j + 200, len(inside) - 1)
                exte = min(200, bd - j)
                if sum(inside[j:bd]) == 0:  # next points are all outside
                    line = numpy.append(line, [poly[j, :]], axis=0)
                    line = numpy.append(line, [poly[j + exte, :]], axis=0)
                    j += exte
                else:  # otherwise keep
                    line = numpy.append(line, [poly[j, :]], axis=0)
            j += 1
        line = numpy.append(line, [[nan, nan]], axis=0)
        out.append(line)
    return _convert_to_array(out)


class Geodata:
    """
    Geographical data class that handles geographical data describing
    shorelines in the form of a shapefile and topobathy in the form of a
    digital elevation model (DEM).
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


def _from_shapefile(filename, bbox):
    """Reads a ESRI Shapefile from `filename` that intersects with `bbox`"""

    polys = []  # tmp storage for polygons and polylines

    print("Reading in shapefile... " + filename)
    s = shapefile.Reader(filename)
    for shape in s.shapes():
        # only read in shapes that intersect with bbox
        if _isOverlapping(bbox, shape.bbox):
            poly = numpy.asarray(shape.points + [(nan, nan)])
            polys.append(poly)

    if len(polys) == 0:
        raise ValueError("Shoreline data does not intersect with bbox")

    return _convert_to_array(polys)


class Shoreline(Geodata):
    """
    The shoreline class extends :class:`Geodata` to store data
    that is later used to create
    signed distance functions to represent irregular
    shoreline geometries.
    """

    def __init__(self, shp, bbox, h0, refinements=1, minimum_area_mult=4.0):
        super().__init__(bbox, h0)

        self.shp = shp
        self.inner = []
        self.outer = []
        self.mainland = []
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult

        @property
        def shp(self):
            return self.__shp

        @shp.setter
        def shp(self, filename):
            if not os.path.isfile(filename):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), filename
                )
            self.__shp = filename

        @property
        def refinements(self):
            return self.__refinements

        @refinements.setter
        def refinements(self, value):
            if value > 0:
                raise ValueError("Refinements must be > 0")
            self.__refinements = value

        @property
        def minimum_area_mult(self, value):
            return self.__minimum_area_mult

        @minimum_area_mult.setter
        def minimum_area_mult(self, value):
            if value <= 0.0:
                raise ValueError(
                    "Minimum area multiplier * h0**2 to "
                    " prune inner geometry must be > 0.0"
                )
            self.__minimum_area_mult = value

        polys = _from_shapefile(self.shp, self.bbox)

        polys = _smoothShoreline(polys, self.refinements)

        polys = _densify(polys, self.h0)

        polys = _nth_simplify(polys, self.bbox)

        self.inner, self.mainland = _classifyShoreline(
            self.bbox, polys, self.h0, self.minimum_area_mult
        )

    def plot(self, hold_on=False):
        """Visualize the content in the shp field of Shoreline"""
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
    """Digitial elevation model read in from a NetCDF file"""

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
