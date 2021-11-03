import errno
import logging
import os

import geopandas as gpd
import matplotlib.path as mpltPath
import numpy as np
import numpy.linalg
import rioxarray as rxr
import shapely.geometry
import shapely.validation
from pyproj import CRS

from .grid import Grid
from .region import Region
from .filterfx import filt2

nan = np.nan

logger = logging.getLogger(__name__)

__all__ = ["Shoreline", "DEM"]


def _convert_to_array(lst):
    """Converts a list of numpy arrays to a np array"""
    return np.concatenate(lst, axis=0)


def _convert_to_list(arr):
    """Converts a nan-delimited numpy array to a list of numpy arrays"""
    a = np.insert(arr, 0, [[nan, nan]], axis=0)
    tmp = [a[s] for s in np.ma.clump_unmasked(np.ma.masked_invalid(a[:, 0]))]
    return [np.append(a, [[nan, nan]], axis=0) for a in tmp]


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
    https://stackoverflow.com/questions/40624409/vectorized-np-linspace-for-multiple-start-and-stop-values
    """
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]


def _densify(poly, maxdiff, bbox, radius=0):
    """Fills in any gaps in latitude or longitude arrays
    that are greater than a `maxdiff` (degrees) apart
    """
    logger.debug("Entering:_densify")

    boubox = _create_boubox(bbox)
    path = mpltPath.Path(boubox, closed=True)
    inside = path.contains_points(poly, radius=0.1)  # add a small radius
    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = np.abs(lat[1:] - lat[:-1])
    dlon = np.abs(lon[1:] - lon[:-1])
    nin = np.ceil(np.maximum(dlat, dlon) / maxdiff) - 1
    nin[~inside[1:]] = 0  # no need to densify outside of bbox please
    # handle negative values
    nin[nin < 0] = 0
    sumnin = np.nansum(nin)
    if sumnin == 0:
        return np.hstack((lon[:, None], lat[:, None]))
    nout = sumnin + nx

    lonout = np.full((int(nout)), nan, dtype=float)
    latout = np.full((int(nout)), nan, dtype=float)

    n = 0
    for i in range(nx - 1):
        ni = nin[i]
        if ni == 0 or np.isnan(ni):
            latout[n] = lat[i]
            lonout[n] = lon[i]
            nstep = 1
        else:
            ni = int(ni)
            icoords = _create_ranges(
                np.array([lat[i], lon[i]]),
                np.array([lat[i + 1], lon[i + 1]]),
                ni + 2,
            )
            latout[n : n + ni + 1] = icoords[0, : ni + 1]
            lonout[n : n + ni + 1] = icoords[1, : ni + 1]
            nstep = ni + 1
        n += nstep

    latout[-1] = lat[-1]
    lonout[-1] = lon[-1]

    logger.debug("Exiting:_densify")

    return np.hstack((lonout[:, None], latout[:, None]))


def _poly_area(x, y):
    """Calculates area of a polygon"""
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _poly_length(coords):
    """Calculates circumference of a polygon"""
    if all(np.isclose(coords[0, :], coords[-1, :])):
        c = coords
    else:
        c = np.vstack((coords, coords[0, :]))

    return np.sum(np.sqrt(np.sum(np.diff(c, axis=0) ** 2, axis=1)))


def _classify_shoreline(bbox, boubox, polys, h0, minimum_area_mult):
    """Classify segments in numpy.array `polys` as either `inner` or `mainland`.
    (1) The `mainland` category contains segments that are not totally enclosed inside the `bbox`.
    (2) The `inner` (i.e., islands) category contains segments totally enclosed inside the `bbox`.
        NB: Removes `inner` geometry with area < `minimum_area_mult`*`h0`**2
    (3) `boubox` polygon array is will be clipped by segments contained by `mainland`.
    """
    logger.debug("Entering:_classify_shoreline")

    _AREAMIN = minimum_area_mult * h0 ** 2

    if len(boubox) == 0:
        boubox = _create_boubox(bbox)
        boubox = np.asarray(boubox)
    elif not _is_path_ccw(boubox):
        boubox = np.flipud(boubox)

    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)

    # Remove nan's (append again at end)
    isNaN = np.sum(np.isnan(boubox), axis=1) > 0
    if any(isNaN):
        boubox = np.delete(boubox, isNaN, axis=0)
    del isNaN

    inner = np.empty(shape=(0, 2))
    inner[:] = nan
    mainland = np.empty(shape=(0, 2))
    mainland[:] = nan

    polyL = _convert_to_list(polys)
    bSGP = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        pSGP = shapely.geometry.Polygon(poly[:-2, :])
        if bSGP.contains(pSGP):
            if pSGP.area >= _AREAMIN:
                inner = np.append(inner, poly, axis=0)
        elif pSGP.overlaps(bSGP):
            # Append polygon segment to mainland
            mainland = np.vstack((mainland, poly))
            # Clip polygon segment from boubox and regenerate path
            bSGP = bSGP.difference(pSGP)

    out = np.empty(shape=(0, 2))

    if bSGP.geom_type == "Polygon":
        bSGP = [bSGP]  # Convert to `MultiPolygon` with 1 member

    # MultiPolygon members can be accessed via iterator protocol using `in`.
    for b in bSGP:
        xy = np.asarray(b.exterior.coords)
        xy = np.vstack((xy, xy[0]))
        out = np.vstack((out, xy, [nan, nan]))

    logger.debug("Exiting:classify_shoreline")

    return inner, mainland, out


def _chaikins_corner_cutting(coords, refinements=5):
    """http://www.cs.unc.edu/~dm/UNC/COMP258/LECTURES/Chaikins-Algorithm.pdf"""
    logger.debug("Entering:_chaikins_corner_cutting")
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    logger.debug("Exiting:_chaikins_corner_cutting")
    return coords


def _smooth_shoreline(polys, N):
    """Smoothes the shoreline segment-by-segment using
    a `N` refinement Chaikins Corner cutting algorithm.
    """
    logger.debug("Entering:_smooth_shoreline")

    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        tmp = _chaikins_corner_cutting(poly[:-1], refinements=N)
        tmp = np.append(tmp, [[nan, nan]], axis=0)
        out.append(tmp)

    logger.debug("Exiting:_smooth_shoreline")

    return _convert_to_array(out)


def _clip_polys_2(polys, bbox, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    """
    logger.debug("Entering:_clip_polys_2")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = np.asarray(_create_boubox(bbox))
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

            logger.info(f"Simplify polygon: length {len(p)} --> {len(p)}")
            # Remove colinear||duplicate vertices
            if len(iRemove) > 0:
                p = np.delete(p, iRemove, axis=0)
                del iRemove

            line = p

            # Close polygon
            if not all(np.isclose(line[0, :], line[-1, :])):
                line = np.append(line, [line[0, :], [nan, nan]], axis=0)
            else:
                line = np.append(line, [[nan, nan]], axis=0)

            out.append(line)

    logger.debug("Exiting:_clip_polys_2")

    return _convert_to_array(out)


def _clip_polys(polys, bbox, delta=0.10):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset `delta` to the `bbox`.
    Dependencies: shapely.geometry and numpy
    """

    logger.debug("Entering:_clip_polys")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    bbox = (bbox[0] - delta, bbox[1] + delta, bbox[2] - delta, bbox[3] + delta)
    boubox = np.asarray(_create_boubox(bbox))
    polyL = _convert_to_list(polys)

    out = np.empty(shape=(0, 2))

    b = shapely.geometry.Polygon(boubox)

    for poly in polyL:
        mp = shapely.geometry.Polygon(poly[:-2, :])
        if not mp.is_valid:
            logger.warning(
                f"polygon {shapely.validation.explain_validity(mp)} Try to make valid."
            )
            mp = mp.buffer(1.0e-5)  # Apply 1 metre buffer
        mp = [mp]

        for p in mp:
            pi = p.intersection(b)
            if b.contains(p):
                out = np.vstack((out, poly))
            elif not pi.is_empty:
                # assert(pi.geom_type,'MultiPolygon')
                if pi.geom_type == "Polygon":
                    pi = [pi]  # `Polygon` -> `MultiPolygon` with 1 member

                for ppi in pi:
                    xy = np.asarray(ppi.exterior.coords)
                    xy = np.vstack((xy, xy[0]))
                    out = np.vstack((out, xy, [nan, nan]))

                del (ppi, xy)
            del pi
        del (p, mp)

    logger.debug("Exiting:_clip_polys")

    return out


def _nth_simplify(polys, bbox):
    """Collapse segments in `polys` outside of `bbox`"""
    logger.debug("Entering:_nth_simplify")

    boubox = np.asarray(_create_boubox(bbox))
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)
    out = []
    for poly in polys:
        j = 0
        inside = path.contains_points(poly[:-2, :])
        line = np.empty(shape=(0, 2))
        while j < len(poly[:-2]):
            if inside[j]:  # keep point (in domain)
                line = np.append(line, [poly[j, :]], axis=0)
            else:  # pt is outside of domain
                bd = min(
                    j + 50, len(inside) - 1
                )  # collapses 50 pts to 1 vertex (arbitary)
                exte = min(50, bd - j)
                if sum(inside[j:bd]) == 0:  # next points are all outside
                    line = np.append(line, [poly[j, :]], axis=0)
                    line = np.append(line, [poly[j + exte, :]], axis=0)
                    j += exte
                else:  # otherwise keep
                    line = np.append(line, [poly[j, :]], axis=0)
            j += 1
        line = np.append(line, [[nan, nan]], axis=0)
        out.append(line)

    logger.debug("Exiting:_nth_simplify")
    return _convert_to_array(out)


def _is_path_ccw(_p):
    """Compute curve orientation from first two line segment of a polygon.
    Source: https://en.wikipedia.org/wiki/Curve_orientation
    """
    detO = 0.0
    O3 = np.ones((3, 3))

    i = 0
    while (i + 3 < _p.shape[0]) and np.isclose(detO, 0.0):
        # Colinear vectors detected. Try again with next 3 indices.
        O3[:, 1:] = _p[i : (i + 3), :]
        detO = np.linalg.det(O3)
        i += 1

    if np.isclose(detO, 0.0):
        raise RuntimeError("Cannot determine orientation from colinear path.")

    return detO > 0.0


def _is_overlapping(bbox1, bbox2):
    """Determines if two axis-aligned boxes intersect"""
    x1min, x1max, y1min, y1max = bbox1
    x2min, x2max, y2min, y2max = bbox2
    return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max


class Shoreline(Region):
    """
    The shoreline class extends :class:`Region` to store data
    that is later used to create signed distance functions to
    represent irregular shoreline geometries.
    """

    def __init__(
        self,
        shp,
        bbox,
        h0,
        crs=4326,
        refinements=1,
        minimum_area_mult=4.0,
        smooth_shoreline=True,
        verbose=1,
    ):
        if isinstance(bbox, tuple):
            _boubox = np.asarray(_create_boubox(bbox))
        else:
            _boubox = np.asarray(bbox)
            if not _is_path_ccw(_boubox):
                _boubox = np.flipud(_boubox)
            bbox = (
                np.nanmin(_boubox[:, 0]),
                np.nanmax(_boubox[:, 0]),
                np.nanmin(_boubox[:, 1]),
                np.nanmax(_boubox[:, 1]),
            )

        super().__init__(bbox, crs)

        self.shp = shp
        self.h0 = h0
        self.inner = []
        self.outer = []
        self.mainland = []
        self.boubox = _boubox
        self.refinements = refinements
        self.minimum_area_mult = minimum_area_mult
        polys = self._read()

        if smooth_shoreline:  # Default, will smooth shoreline
            polys = _smooth_shoreline(polys, self.refinements)

        polys = _densify(polys, self.h0, self.bbox)

        polys = _clip_polys(polys, self.bbox)

        self.inner, self.mainland, self.boubox = _classify_shoreline(
            self.bbox, self.boubox, polys, self.h0 / 2, self.minimum_area_mult
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
        self.__h0 = value

    @staticmethod
    def transform_to(gdf, dst_crs):
        """Transform geodataframe ``gdf`` representing
        a shoreline to dst_crs
        """
        dst_crs = CRS.from_user_input(dst_crs)
        if not gdf.crs.equals(dst_crs):
            logger.info(f"Reprojecting shoreline from {gdf.crs} to {dst_crs}")
            gdf = gdf.to_crs(dst_crs)
        return gdf

    def _read(self):
        """Reads a ESRI Shapefile from `filename` ∩ `bbox`"""
        if not isinstance(self.bbox, tuple):
            _bbox = (
                np.amin(self.bbox[:, 0]),
                np.amax(self.bbox[:, 0]),
                np.amin(self.bbox[:, 1]),
                np.amax(self.bbox[:, 1]),
            )
        else:
            _bbox = self.bbox

        logger.debug("Entering: _read")

        msg = f"Reading in ESRI Shapefile {self.shp}"
        logger.info(msg)

        # transform if necessary
        s = self.transform_to(gpd.read_file(self.shp), self.crs)

        polys = []  # store polygons

        delimiter = np.empty((1, 2))
        delimiter[:] = np.nan
        re = numpy.array([0, 2, 1, 3], dtype=int)
        for g in s.geometry:
            # extent of geometry
            bbox2 = [g.bounds[r] for r in re]
            if _is_overlapping(_bbox, bbox2):
                poly = np.asarray(g.exterior.coords.xy).T
                polys.append(np.row_stack((poly, delimiter)))

        if len(polys) == 0:
            raise ValueError("Shoreline data does not intersect with bbox")

        logger.debug("Exiting: _read")

        return _convert_to_array(polys)

    def plot(
        self,
        ax=None,
        xlabel=None,
        ylabel=None,
        title=None,
        file_name=None,
        show=True,
        xlim=None,
        ylim=None,
        loc=None,
    ):
        """Visualize the content in the shp field of Shoreline"""
        import matplotlib.pyplot as plt

        flg1, flg2 = False, False

        if ax is None:
            fig, ax = plt.subplots()
            ax.axis("equal")

        if len(self.mainland) != 0:
            (line1,) = ax.plot(self.mainland[:, 0], self.mainland[:, 1], "k-")
            flg1 = True
        if len(self.inner) != 0:
            (line2,) = ax.plot(self.inner[:, 0], self.inner[:, 1], "r-")
            flg2 = True
        (line3,) = ax.plot(self.boubox[:, 0], self.boubox[:, 1], "g--")

        xmin, xmax, ymin, ymax = self.bbox
        rect = plt.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=None,
            hatch="////",
            alpha=0.2
        )

        border = 0.10 * (xmax - xmin)
        loc_ = 'best' if loc is None else loc
        if ax is None:
            plt.xlim(xmin - border, xmax + border)
            plt.ylim(ymin - border, ymax + border)

        ax.add_patch(rect)

        if flg1 and flg2:
            ax.legend((line1, line2, line3), ("mainland", "inner", "outer"), loc=loc_)
        elif flg1 and not flg2:
            ax.legend((line1, line3), ("mainland", "outer"), loc=loc_)
        elif flg2 and not flg1:
            ax.legend((line2, line3), ("inner", "outer"), loc=loc_)

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


class DEM(Grid):
    """Digitial elevation model read in from a tif or NetCDF file
    parent class is a :class:`Grid`
    """

    def __init__(self, dem, crs=4326, bbox=None):
        if type(dem) == str:
            basename, ext = os.path.splitext(dem)
            if ext.lower() in [".nc"] or [".tif"]:
                topobathy, reso, bbox = self._read(dem, bbox, crs)
                topobathy = topobathy.astype(float)

            else:
                raise ValueError(f"DEM file {dem} has unknown format {ext[1:]}.")

            self.dem = dem

        elif type(dem) == numpy.ndarray:
            topobathy = dem.astype(float)
            reso = ((bbox[1]-bbox[0])/topobathy.shape[0], (bbox[3]-bbox[2])/topobathy.shape[1])
            self.dem = 'input'

        elif callable(dem): # if input is a function
            lon, lat = np.linspace(bbox[0], bbox[1], 1001), np.linspace(bbox[2], bbox[3], 1001)
            reso = ((bbox[1]-bbox[0])/lon.shape[0], (bbox[3]-bbox[2])/lat.shape[0])
            lon, lat = np.meshgrid(lon, lat)
            topobathy = dem(lon, lat)
            self.dem = 'function'

        topobathy[abs(topobathy) > 1e5] = np.NaN

        super().__init__(
            bbox=bbox,
            crs=crs,
            dx=reso[0],
            dy=np.abs(reso[1]),
            values=np.rot90(topobathy, 1),
        )
        super().build_interpolant()

    @staticmethod
    def transform_to(xry, dst_crs):
        """Transform xarray ``xry`` representing
        a raster to dst_crs
        """
        dst_crs = CRS.from_user_input(dst_crs)
        if not xry.crs.equals(dst_crs):
            msg = f"Reprojecting raster from {xry.crs} to {dst_crs}"
            logger.debug(msg)
            xry = xry.rio.reproject(dst_crs)
        return xry

    def _read(self, filename, bbox, crs):
        """Read in a digitial elevation model from a NetCDF or GeoTif file"""
        logger.debug("Entering: DEM._read")

        msg = f"Reading in {filename}"
        logger.info(msg)

        with rxr.open_rasterio(filename) as r:
            # warp/reproject it if necessary
            r = self.transform_to(r, crs)
            # entire DEM is read in
            if bbox is None:
                bnds = r.rio.bounds()
                bbox = (bnds[0], bnds[2], bnds[1], bnds[3])
            else:
                # then we clip the DEM to the box
                r = r.rio.clip_box(
                    minx=bbox[0], miny=bbox[2], maxx=bbox[1], maxy=bbox[3]
                )
            topobathy = r.data[0]

            logger.debug("Exiting: DEM._read")
            return topobathy, r.rio.resolution(), bbox

    def filter_bathymetry(self, fl, min_depth=50):
        lon, lat = self.create_grid()
        tmpz = abs(self.eval((lon, lat)))
        tmpz[tmpz < min_depth] = min_depth
        tmpz[tmpz > 5e5] = nan

        tmpz_f = np.zeros(tmpz.shape)
        rbfilt = None

        # Now filtering bathymetry to obtain only relevant features
        # Loop through each set of bandpass filter lengths
        if fl[0] < 0 and fl[0] != -999:
            logger.info("Rossby radius of deformation filter is on.")
            rbfilt = abs(fl[0])
            fl = []
            filtit = True

        elif fl[0] == 0:
            logger.info("Slope filter is off.")
            fl = []
            tmpz_f[:] = tmpz[:]
            filtit = False

        elif fl[0] == -999:
            logger.info("Slope filter is LEGACY.")
            fl = []
            tmpz_f[:] = tmpz[:]
            filtit = -999

        else:
            filtit = False
            for slp in fl.T:
                if np.isscalar(slp):
                    # Do a low-pass filter
                    tmpz[tmpz > 3000] = np.NaN #Handling NaN values
                    tmpz_ft = filt2(tmpz, self.dy, slp, "lp")

                elif slp[1] == 0:
                    tmpz_ft = filt2(tmpz, self.dy, slp[0], "lp")

                elif np.all(slp != 0):
                    # Do a bandpass filter
                    tmpz_ft = filt2(tmpz, self.dy, slp, "bp")

                else:
                    # Highpass filter not recommended
                    print(
                        "Warning:: Highpass filter on bathymetry in slope - \
        edgelength function in not recommended"
                    )
                    print('Warning')
                    tmpz_ft = filt2(tmpz, self.dy, slp[1], "hp")

                tmpz_f += tmpz_ft
        print('done?')
        return tmpz_f, fl, filtit, rbfilt

    def grad_bathymetric_filter(self, tmpz, fl, filtit, rbfilt, barot):

        # Performs bandpass filtering on Rossby radius of deformation
        if filtit:
            bs = self.rossby_filter(
                tmpz, rbfilt, barot
            )

            # legacy filter
        elif filtit == -999:
            bs = self.legacy_filter(tmpz)

        else:
            # get slope from (possibly filtered) bathymtry (without having
            # by the localised Rossby radius of deformation filtered)

            # Perform bathymetric gradient
            by, bx = EarthGradient(tmpz, self.dy, self.dx)

            #Calulates the modulus of bathymetric gradient (a measure of slope)
            bs = np.sqrt(bx ** 2 + by ** 2)


        return bs


    def rossby_filter(self, tmpz, rbfilt, barot):
        """
        Performs the Rossby radius filtering if filtit==True in
        slope_sizing_function.

        Parameters
        ----------
        tmpz : numpy.ndarray
            Contains the bathymetric data across the grid formed by coordinate
            arrays (xg, yg).
        bbox : tuple
            Describes the boundary box of our domain.
        grid_details : tuple
            Contains the information regarding normals and grid resolutions,
            (nx, ny, dx, dy).
        coords : tuple np.ndarray
            A tuple of two numpy.ndarray describing the longitude and latitude
            coordinate system of our grid.
        rbfilt : float
            Describes the corresponding rossby radius to filter out
        barot : bool
            If True, the function uses the barotropic Rossby radius of deformation.

        Returns
        -------
        bs : numpy.ndarray
            This is essentially grad(h) squared after performing the bandpass
            filtering on the Rossby radius of deformation.
        time_taken : float
            the time taken to prform the filtering process.

        """
        import time
        import math

        x0, xN, y0, yN = self.bbox
        xg, yg = self.create_grid()

        start = time.perf_counter()
        bs = np.empty(tmpz.shape)
        bs[:] = np.nan

        # Break into 10 deg latitude chuncsm or less if higher resolution
        div = math.ceil(min(1e7 / self.nx, 10 * self.ny / (yN - y0)))
        grav, Rre = 9.807, 7.29e-5  # Gravity and Rotation rate of Earth in radians
        # per second
        print('div, ny', div, self.ny)
        nb = math.ceil(self.ny / div)
        n2s = 0
        dx = self.dy * np.cos(np.pi * np.minimum(yg[0, :], 85) / 180)

        for jj in range(nb):
            n2e = min(self.ny, n2s + div)
            # Rossby radius of deformation filter
            # See Shelton, D. B., et al. (1998): Geographical variability of the
            # first-baroclinic Rossby radius of deformation. J. Phys. Oceanogr.,
            # 28, 433-460.
            ygg = yg[:, n2s:n2e+1]
            dxx = np.mean(dx[n2s:n2e+1])
            f = 2 * Rre * abs(np.sin(ygg * np.pi / 180))
            if barot:
                # Barotropic case
                c = np.sqrt(grav * np.maximum(1, -tmpz[:, n2s:n2e+1]))

            else:
                # Baroclinic case (estimate Nm to be 2.5e-3)
                Nm = 2.5e-3  # Δz x N, where N is Brunt-Vaisala frequency,
                # sqrt(-g/ρ0 * dρ/dz), giving sqrt(-g * (Δρ/ρ0) * Δz)
                c = Nm * np.maximum(1, -tmpz[:, n2s:n2e+1]) / np.pi

            rosb = c / f
            # Update for equatorial regions
            indices = abs(ygg) < 5
            Re = 6.371e6  # Earth radius at equator in SI units of metres
            twobeta = 4 * Rre * np.cos(ygg[indices] * np.pi / 180) / Re
            rosb[indices] = np.sqrt(c[indices] / twobeta)
            # limit rossby radius to 10,000 km for practical purposes
            rosb[rosb > 1e7] = 1e7
            # Keep lengthscales rbfilt * barotropic
            # radius of deformation
            rosb = np.minimum(10, np.maximum(0, np.floor(np.log2(rosb / \
                                                        self.dy / rbfilt))))
            edges = np.unique(np.copy(rosb))
            bst = rosb * 0
            for i in range(len(edges)):
                if edges[i] > 0:
                    mult = 2 ** edges[i]
                    xl, xu = 1, self.nx
                    if ((np.max(xg) > 179 and np.min(xg) < -179)) or (
                        np.max(xg) > 359 and np.min(xg) < 1
                    ):
                        # wraps around
                        logger.info("wrapping around")
                        xr = np.concatenate(
                            [
                                np.arange(self.nx - mult / 2, self.nx),
                                np.arange(xl, xu+1),
                                np.arange(1, mult / 2),
                            ],
                            dtype=int,
                        )
                    else:
                        xr = np.arange(xl, xu+1, dtype=int)

                    yl, yu = max(1, n2s - mult / 2), min(self.ny, n2e + mult / 2)
                    if np.max(yg) > 89 and yu == self.ny:
                        # create mirror around pole
                        yr = np.concatenate(
                            [
                                np.arange(yl, yu+1),
                                np.arange(yu - 1, 1 + 2 * self.ny - n2e - mult / 2, -1),
                            ],
                            dtype=int,
                        )
                    else:
                        yr = np.arange(yl, yu+1, dtype=int)

                    xr, yr = xr[:, None]-1, yr[None, :]-1

                    if mult == 2:
                        tmpz_ft = filt2(tmpz[xr, yr], min([self.dx, self.dy]), self.dy * 2.01, "lp")
                    else:
                        tmpz_ft = filt2(tmpz[xr, yr], min([self.dx, self.dy]), self.dy * mult, "lp")

                    # delete the padded region
                    #tmpz_ft[: 1+np.where(xr == 0)[0][0], :] = 0
                    #tmpz_ft[self.nx:, :] = 0
                    #tmpz_ft[:, :1+np.where(yr == n2s)[0][0]] = 0
                    tmpz_ft = tmpz[:, n2s:n2e+1]

                else:
                    tmpz_ft = tmpz[:, n2s:n2e+1]

                by, bx = EarthGradient(
                    tmpz_ft, self.dy, self.dx
                ) # bathymetric gradient
                tempbs = np.sqrt(bx ** 2 + by ** 2)  #modulus of bathymetric gradient (slope)
                print(bst.shape, tempbs.shape, rosb.shape, edges.shape, i)
                print((rosb==edges[i]).shape)
                bst[rosb == edges[i]] = tempbs[rosb == edges[i]]

            bs[:, n2s:n2e+1] = bst
            n2s = n2e

        time_taken = time.perf_counter() - start
        logger.info(f"It took {time_taken} seconds to perform filtering on\
Rossby radius of deformation")

        return bs

    def legacy_filter(self, tmpz):
        """
        Calculates the modulus of the bathymetric gradient having performed a
        simple filter on the Rossby radius of deformation (but not localised)

        Parameters
        ----------
        tmpz : numpy.ndarray
            Bathymetric data

        Returns
        -------
        bs : numpy.ndarray
            Modulus of bathymetric gradient (a measure of topographic slope)
            having performed a filter on the Rossby radius of deformation,
            if specified.

        """
        from math import sqrt
        from filterfx import filt2

        x0, xN, y0, yN = self.bbox
        yg = np.arange(y0, yN+self.dy, self.dy)

        grav, Rre = 9.807, 7.29e-5  # Gravity and Rotation rate of Earth in radians
        # per second

        bs = np.empty((self.nx, self.ny))
        bs[:] = np.nan
        # Rossby radius of deformation filter
        f = 2 * Rre * abs(np.sin(yg * np.pi / 180))  # Local Coriolis coefficient
        # limit to 1000 km
        rosb = np.minimum(
            1000e3, sqrt(grav * abs(tmpz)) / f
        )  # Gives local Rossby radius everywhere
        # autmatically divide into discrete bins
        _, edges = np.histogram(rosb)
        tmpz_ft = tmpz
        dyb = self.dy
        # get slope from filtered bathy for the segment only
        by, bx = EarthGradient(tmpz_ft, self.dy, self.dx)  # get slope in x and y directions
        tempbs = np.sqrt(bx ** 2 + by ** 2) #performs the modulus of bathymetric gradient
        # get overall slope
        for i in range(len(edges) - 1):
            sel = (rosb >= edges[i]) & (rosb <= edges[i + 1])
            rosbylb = np.mean(edges[i : i + 1])

            if rosbylb > 2 * dyb:
                tmpz_ft = filt2(tmpz_ft, dyb, rosbylb, "lp")
                dyb = rosbylb

                # get slope from filtered bathy for the segment only
                by, bx = EarthGradient(tmpz_ft, self.dy, self.dx)  # get slope in x and y directions
                tempbs = np.sqrt(bx ** 2 + by ** 2)  # #updates modulus of bathymetric gradient

            else:
                # otherwise just use the same tempbs from before
                pass

            # put in the full one
            bs[sel] = tempbs[sel]

        return bs


def EarthGradient(F, dy, dx):
    """
    EarthGradient(F,HX,HY), where F is 2-D, uses the spacing
    specified by HX and HY. HX and HY can either be scalars to specify
    the spacing between coordinates or vectors to specify the
    coordinates of the points.  If HX and HY are vectors, their length
    must match the corresponding dimension of F.
    """
    Fy, Fx = np.zeros(F.shape), np.zeros(F.shape)

    # Forward diferences on edges
    Fx[:, 0] = (F[:, 1] - F[:, 0]) / dx
    Fx[:, -1] = (F[:, -1] - F[:, -2]) / dx
    Fy[0, :] = (F[1, :] - F[0, :]) / dy
    Fy[-1, :] = (F[-1, :] - F[-2, :]) / dy

    # Central Differences on interior
    Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dx)
    Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dy)

    return Fy, Fx
