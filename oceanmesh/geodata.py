import errno
import os
from math import ceil, floor

import matplotlib.path as mpltPath
import numpy
import numpy.linalg
import rasterio
import shapefile
import shapely.geometry
import shapely.validation
# from affine import Affine
from pyproj import Proj
from rasterio.windows import Window

from .grid import Grid

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
    """Create a bounding box from domain extents `bbox`. Path orientation will be CCW."""
    if isinstance(bbox, tuple):
        xmin, xmax, ymin, ymax = bbox
        return [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ]
    return bbox


def _create_ranges(start, stop, N, endpoint=True):
    """Vectorized alternative to numpy.linspace
    https://stackoverflow.com/questions/40624409/vectorized-numpy-linspace-for-multiple-start-and-stop-values
    """
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * numpy.arange(N) + start[:, None]


def _densify(poly, maxdiff, bbox, radius=0):
    """Fills in any gaps in latitude or longitude arrays
    that are greater than a `maxdiff` (degrees) apart
    """
    boubox = _create_boubox(bbox)
    path = mpltPath.Path(boubox, closed=True)
    inside = path.contains_points(poly, radius=0.1)  # add a small radius
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = numpy.abs(lat[1:] - lat[:-1])
    dlon = numpy.abs(lon[1:] - lon[:-1])
    nin = numpy.ceil(numpy.maximum(dlat, dlon) / maxdiff) - 1
    nin[~inside[1:]] = 0  # no need to densify outside of bbox please
    # handle negative values
    nin[nin < 0] = 0
    sumnin = numpy.nansum(nin)
    if sumnin == 0:
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


def _poly_area(x, y):
    """Calculates area of a polygon"""
    return 0.5 * numpy.abs(
        numpy.dot(x, numpy.roll(y, 1)) - numpy.dot(y, numpy.roll(x, 1))
    )


def _poly_length(coords):
    """Calculates circumference of a polygon"""
    if all(numpy.isclose(coords[0, :], coords[-1, :])):
        c = coords
    else:
        c = numpy.vstack((coords, coords[0, :]))

    return numpy.sum(numpy.sqrt(numpy.sum(numpy.diff(c, axis=0) ** 2, axis=1)))


def _is_overlapping(bbox1, bbox2):
    """Determines if two axis-aligned boxes intersect"""
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


def _classify_shoreline(bbox, boubox, polys, h0, minimum_area_mult, verbose):
    """Classify segments in numpy.array `polys` as either `inner` or `mainland`.
    (1) The `mainland` category contains segments that are not totally enclosed inside the `bbox`.
    (2) The `inner` (i.e., islands) category contains segments totally enclosed inside the `bbox`.
        NB: Removes `inner` geometry with area < `minimum_area_mult`*`h0`**2
    (3) `boubox` polygon array is will be clipped by segments contained by `mainland`.
    """
    if verbose > 1:
        print("Classifying shoreline segments...")

    _AREAMIN = minimum_area_mult * h0 ** 2

    if len(boubox) == 0:
        boubox = _create_boubox(bbox)
        boubox = numpy.asarray(boubox)
    elif not _is_path_ccw(boubox):
        boubox = numpy.flipud(boubox)

    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)

    # Remove nan's (append again at end)
    isNaN = numpy.sum(numpy.isnan(boubox), axis=1) > 0
    if any(isNaN):
        boubox = numpy.delete(boubox, isNaN, axis=0)
    del isNaN

    inner = numpy.empty(shape=(0, 2))
    inner[:] = nan
    mainland = numpy.empty(shape=(0, 2))
    mainland[:] = nan

    polyL = _convert_to_list(polys)
    bSGP = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        pSGP = shapely.geometry.Polygon(poly[:-2, :])
        if bSGP.contains(pSGP):
            if pSGP.area >= _AREAMIN:
                inner = numpy.append(inner, poly, axis=0)
        elif pSGP.overlaps(bSGP):
            # Append polygon segment to mainland
            mainland = numpy.vstack((mainland, poly))
            # Clip polygon segment from boubox and regenerate path
            bSGP = bSGP.difference(pSGP)

    out = numpy.empty(shape=(0, 2))

    if bSGP.geom_type == "Polygon":
        bSGP = [bSGP]  # Convert to `MultiPolygon` with 1 member

    # MultiPolygon members can be accessed via iterator protocol using `in`.
    for b in bSGP:
        xy = numpy.asarray(b.exterior.coords)
        xy = numpy.vstack((xy, xy[0]))
        out = numpy.vstack((out, xy, [nan, nan]))

    return inner, mainland, out


def _chaikins_corner_cutting(coords, refinements=5):
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


def _smooth_shoreline(polys, N, verbose):
    """Smoothes the shoreline segment-by-segment using
    a `N` refinement Chaikins Corner cutting algorithm.
    """
    if verbose > 1:
        print("Smoothing shoreline segments...")
    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        tmp = _chaikins_corner_cutting(poly[:-1], refinements=N)
        tmp = numpy.append(tmp, [[nan, nan]], axis=0)
        out.append(tmp)
    return _convert_to_array(out)


def _clip_polys_2(polys, bbox, verbose, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    """
    if verbose > 1:
        print("Collapsing polygon segments outside bbox...")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = numpy.asarray(_create_boubox(bbox))
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)

    out = []

    for poly in polys:
        p = poly[:-1, :]

        inside = path.contains_points(p)

        iRemove = []

        _keepLL = True
        _keepUL = True
        _keepLR = True
        _keepUR = True

        if all(inside):
            out.append(poly)
        elif any(inside):
            for j in range(0, len(p)):
                if not inside[j]:  # snap point to inflated domain bounding box
                    px = p[j, 0]
                    py = p[j, 1]
                    if not (bbox[0] < px and px < bbox[1]) or not (
                        bbox[2] < py and py < bbox[3]
                    ):
                        if (
                            _keepLL and px < bbox[0] and py < bbox[2]
                        ):  # is over lower-left
                            p[j, :] = [bbox[0], bbox[2]]
                            _keepLL = False
                        elif (
                            _keepUL and px < bbox[0] and bbox[3] < py
                        ):  # is over upper-left
                            p[j, :] = [bbox[0], bbox[3]]
                            _keepUL = False
                        elif (
                            _keepLR and bbox[1] < px and py < bbox[2]
                        ):  # is over lower-right
                            p[j, :] = [bbox[1], bbox[2]]
                            _keepLR = False
                        elif (
                            _keepUR and bbox[1] < px and bbox[3] < py
                        ):  # is over upper-right
                            p[j, :] = [bbox[1], bbox[3]]
                            _keepUR = False
                        else:
                            iRemove.append(j)

            # print('Simplify polygon: length {:d} --> {:d}'.format(len(p),len(p)-len(iRemove)))
            # Remove colinear||duplicate vertices
            if len(iRemove) > 0:
                p = numpy.delete(p, iRemove, axis=0)
                del iRemove

            line = p

            # Close polygon
            if not all(numpy.isclose(line[0, :], line[-1, :])):
                line = numpy.append(line, [line[0, :], [nan, nan]], axis=0)
            else:
                line = numpy.append(line, [[nan, nan]], axis=0)

            out.append(line)

    return _convert_to_array(out)


def _clip_polys(polys, bbox, verbose, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    Dependencies: shapely.geometry and numpy
    """

    if verbose > 1:
        print("Collapsing polygon segments outside bbox...")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = numpy.asarray(_create_boubox(bbox))
    polyL = _convert_to_list(polys)

    out = numpy.empty(shape=(0, 2))

    b = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        mp = shapely.geometry.Polygon(poly[:-2, :])
        if mp.is_valid:
            mp = [mp]
        else:
            if verbose > 0:
                print(
                    "Warning, polygon",
                    shapely.validation.explain_validity(mp),
                    "Try to make valid.",
                )
            mp = mp.buffer(1.0e-5)  # Apply 1 metre buffer
        for p in mp:
            pi = p.intersection(b)
            if b.contains(p):
                out = numpy.vstack((out, poly))
            elif not pi.is_empty:
                # assert(pi.geom_type,'MultiPolygon')
                if pi.geom_type == "Polygon":
                    pi = [pi]  # `Polygon` -> `MultiPolygon` with 1 member

                for ppi in pi:
                    xy = numpy.asarray(ppi.exterior.coords)
                    xy = numpy.vstack((xy, xy[0]))
                    out = numpy.vstack((out, xy, [nan, nan]))

                del (ppi, xy)
            del pi
        del (p, mp)

    return out


def _nth_simplify(polys, bbox, verbose):
    """Collapse segments in `polys` outside of `bbox`"""
    if verbose > 1:
        print("Collapsing segments outside bbox...")
    boubox = numpy.asarray(_create_boubox(bbox))
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
                bd = min(
                    j + 50, len(inside) - 1
                )  # collapses 50 pts to 1 vertex (arbitary)
                exte = min(50, bd - j)
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


def _is_path_ccw(_p):
    """Compute curve orientation from first two line segment of a polygon.
    Source: https://en.wikipedia.org/wiki/Curve_orientation
    """
    detO = 0.0
    O3 = numpy.ones((3, 3))

    i = 0
    while (i + 3 < _p.shape[0]) and numpy.isclose(detO, 0.0):
        # Colinear vectors detected. Try again with next 3 indices.
        O3[:, 1:] = _p[i : (i + 3), :]
        detO = numpy.linalg.det(O3)
        i += 1

    if numpy.isclose(detO, 0.0):
        raise RuntimeError("Cannot determine orientation from colinear path.")

    return detO > 0.0


class Geodata:
    """
    Geographical data class that handles geographical data describing
    shorelines in the form of a shapefile and topobathy in the form of a
    digital elevation model (DEM).
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
            if isinstance(value, tuple):
                if len(value) < 4:
                    raise ValueError("bbox has wrong number of values.")
                if value[1] < value[0]:
                    raise ValueError("bbox has wrong values.")
                if value[3] < value[2]:
                    raise ValueError("bbox has wrong values.")
            self.__bbox = value


def _from_shapefile(filename, bbox, verbose):
    """Reads a ESRI Shapefile from `filename` ∩ `bbox`"""
    if not isinstance(bbox, tuple):
        bbox = (
            numpy.amin(bbox[:, 0]),
            numpy.amax(bbox[:, 0]),
            numpy.amin(bbox[:, 1]),
            numpy.amax(bbox[:, 1]),
        )
    polys = []  # tmp storage for polygons and polylines

    if verbose > 0:
        print("Reading in ESRI Shapefile... " + filename)
    s = shapefile.Reader(filename)
    re = numpy.array([0, 2, 1, 3], dtype=int)
    for shape in s.shapes():
        # only read in shapes that intersect with bbox
        bbox2 = [shape.bbox[r] for r in re]
        if _is_overlapping(bbox, bbox2):
            poly = numpy.asarray(shape.points + [(nan, nan)])
            polys.append(poly)

    if len(polys) == 0:
        raise ValueError("Shoreline data does not intersect with bbox")

    return _convert_to_array(polys)


class Shoreline(Geodata):
    """
    The shoreline class extends :class:`Geodata` to store data
    that is later used to create signed distance functions to
    represent irregular shoreline geometries.
    """

    def __init__(self, shp, bbox, h0, refinements=1, minimum_area_mult=4.0, verbose=1):

        if isinstance(bbox, tuple):
            _boubox = numpy.asarray(_create_boubox(bbox))
        else:
            _boubox = numpy.asarray(bbox)
            if not _is_path_ccw(_boubox):
                _boubox = numpy.flipud(_boubox)
            bbox = (
                numpy.nanmin(_boubox[:, 0]),
                numpy.nanmax(_boubox[:, 0]),
                numpy.nanmin(_boubox[:, 1]),
                numpy.nanmax(_boubox[:, 1]),
            )

        super().__init__(bbox)

        self.shp = shp
        self.h0 = h0  # this converts meters -> wgs84 degees
        self.inner = []
        self.outer = []
        self.mainland = []
        self.boubox = _boubox
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult

        polys = _from_shapefile(self.shp, self.bbox, verbose)

        polys = _smooth_shoreline(polys, self.refinements, verbose)

        polys = _densify(polys, self.h0, self.bbox, verbose)

        polys = _clip_polys(polys, self.bbox, verbose)

        self.inner, self.mainland, self.boubox = _classify_shoreline(
            self.bbox, self.boubox, polys, self.h0 / 2, self.minimum_area_mult, verbose
        )

    @property
    def shp(self):
        return self.__shp

    @shp.setter
    def shp(self, filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), filename)
        self.__shp = filename

    @property
    def refinements(self):
        return self.__refinements

    @refinements.setter
    def refinements(self, value):
        if value < 0:
            raise ValueError("Refinements must be > 0")
        self.__refinements = value

    @property
    def minimum_area_mult(self):
        return self.__minimum_area_mult

    @minimum_area_mult.setter
    def minimum_area_mult(self, value):
        if value <= 0.0:
            raise ValueError(
                "Minimum area multiplier * h0**2 to "
                " prune inner geometry must be > 0.0"
            )
        self.__minimum_area_mult = value

    @property
    def h0(self):
        return self.__h0

    @h0.setter
    def h0(self, value):
        if value <= 0:
            raise ValueError("h0 must be > 0")
        value /= 111e3  # convert to wgs84 degrees
        self.__h0 = value

    def plot(
        self,
        ax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        file_name=None,
        show=True,
    ):
        """Visualize the content in the shp field of Shoreline"""
        import matplotlib.pyplot as plt

        flg1, flg2 = False, False

        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")

        if len(self.mainland) != 0:
            (line1,) = ax.plot(self.mainland[:, 0], self.mainland[:, 1], "kx-")
            flg1 = True
        if len(self.inner) != 0:
            (line2,) = ax.plot(self.inner[:, 0], self.inner[:, 1], "rx-")
            flg2 = True
        (line3,) = ax.plot(self.boubox[:, 0], self.boubox[:, 1], "gx-")

        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=None,
            hatch="///",
            alpha=0.2,
        )

        border = 0.10 * (xmax - xmin)
        plt.xlim(xmin - border, xmax + border)
        plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        if flg1 and flg2:
            ax.legend((line1, line2, line3), ("mainland", "inner", "outer"))
        elif flg1 and not flg2:
            ax.legend((line1, line3), ("mainland", "outer"))
        elif flg2 and not flg1:
            ax.legend((line2, line3), ("inner", "outer"))

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        ax.set_aspect("equal", adjustable="box")

        if show:
            plt.show()
        if file_name is not None:
            plt.savefig(file_name)
        return ax


def _longlat2window(lon, lat, dataset):
    """
    Args:
        lon (tuple): Tuple of min and max lon
        lat (tuple): Tuple of min and max lat
        dataset: Rasterio dataset

    Returns:
        rasterio.windows.Window
    """
    p = Proj(dataset.crs)
    t = dataset.transform
    xmin, ymin = p(lon[0], lat[0])
    xmax, ymax = p(lon[1], lat[1])
    col_min, row_min = ~t * (xmin, ymin)
    col_max, row_max = ~t * (xmax, ymax)
    return Window.from_slices(
        rows=(floor(row_max), ceil(row_min)), cols=(floor(col_min), ceil(col_max))
    )


def _from_file(filename, bbox, verbose):
    """Read in a digitial elevation model from a NetCDF or GeoTif file"""

    if verbose:
        print(f"Reading in {filename}")

    with rasterio.open(filename) as r:

        if bbox is None:
            bbox = (
                r.bounds.left,
                r.bounds.right,
                r.bounds.bottom,
                r.bounds.top,
            )
        window = _longlat2window((bbox[0], bbox[1]), (bbox[2], bbox[3]), r)
        topobathy = r.read(1, window=window)
        reso = r.res
        # This creates coordinate grids
        # T0 = r.transform  # upper-left pixel corner affine transform
        # T1 = T0 * Affine.translation(0.5, 0.5)
        # cols, rows = numpy.meshgrid(
        #    numpy.arange(topobathy.shape[2]), numpy.arange(topobathy.shape[1])
        # )
        ## Function to convert pixel row/column index (from 0) to easting/northing at centre
        # def rc2en(r, c):
        #    return (c, r) * T1

    return topobathy, reso, bbox


class DEM(Grid):
    """Digitial elevation model read in from a tif or NetCDF file
    parent class is a :class:`Grid`
    """

    def __init__(self, dem, bbox=None, verbose=1):

        basename, ext = os.path.splitext(dem)
        if ext.lower() in [".nc"] or [".tif"]:
            topobathy, reso, bbox = _from_file(dem, bbox, verbose)
        else:
            raise ValueError(f"DEM file {dem} has unknown format {ext[1:]}.")

        self.dem = dem
        super().__init__(
            bbox=bbox, dx=reso[0], dy=reso[1], values=numpy.rot90(topobathy, 3)
        )
        super().build_interpolant()
