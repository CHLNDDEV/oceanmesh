import errno
import os

import matplotlib.path as mpltPath
import numpy
import numpy.linalg
import shapefile
from netCDF4 import Dataset
from PIL import Image
from PIL.TiffTags import TAGS

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
    """Create a bounding box from domain extents `bbox`."""
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
    """Vectorized alternative to numpy.linspace"""
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
    path = mpltPath.Path(boubox)
    inside = path.contains_points(poly, radius=radius)

    lon, lat = poly[:, 0], poly[:, 1]
    nx = len(lon)
    dlat = numpy.abs(lat[1:] - lat[:-1])
    dlon = numpy.abs(lon[1:] - lon[:-1])
    nin = numpy.ceil(numpy.maximum(dlat, dlon) / maxdiff) - 1
    nin[~inside[1:]] = 0  # no need to densify outside of bbox please
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
    if all(numpy.isclose(coords[0,:],coords[-1,:])):
      c = coords
    else:
      c = numpy.vstack((coords,coords[0,:]))

    return numpy.sum(numpy.sqrt(numpy.sum(numpy.diff(c,axis=0)**2,axis=1)))

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

    if boubox == []:
      boubox = _create_boubox(bbox)
      boubox = numpy.asarray(boubox)

    if not _is_path_ccw(boubox):
      boubox = numpy.flipud(boubox)

    boubox = _densify(boubox, h0 / 2, bbox, radius=0.1)
  # Remove nan's (append again at end)
    isNaN = ( numpy.sum(numpy.isnan(boubox),axis=1)>0 )
    if any(isNaN):
      boubox = numpy.delete(boubox,isNaN,axis=0)
    del(isNaN)

    inner = numpy.empty(shape=(0, 2))
    inner[:] = nan
    mainland = numpy.empty(shape=(0, 2))
    mainland[:] = nan

    polys = _convert_to_list(polys)
    path = mpltPath.Path(boubox)
    for poly in polys:
      inside = path.contains_points(poly[:-2])
      if all(inside):
        area = _poly_area(poly[:-2, 0], poly[:-2, 1])
        if area < minimum_area_mult * h0 ** 2:
          continue
        inner = numpy.append(inner, poly, axis=0)
      elif any(inside):
      # Append polygon segment to mainland
        mainland = numpy.vstack((mainland,poly))
      # Clip polygon segment from boubox and regenerate path
        outside = mpltPath.Path(poly[:-2,:]).contains_points(boubox)
        if any(outside):
        # 1/4 Remove points outside from boubox
          boubox = boubox[~outside,:]
        # 2/4 Create a list of segments that are inside boubox (segment to add)
          _isegs = _index_segments_to_list(numpy.where(inside)[0])
        # 3/4 Loop though segments and find insertion indexes
          numpy.set_printoptions(precision=3,suppress=True)
          for _iseg in _isegs:
            _n = len(_iseg)
            _pi = poly[:-2,:][_iseg,:]
            i,j = _find_closest_index(boubox,_pi)

          # Because `outside` was removed from path, indexes i,j are expected
          # to be neighbours. Make case for last to first index.
            if abs(j-1)!=1 and j<i:
              _pi = numpy.flipud(_pi)
              i,j = _find_closest_index(boubox,_pi)

            if i<j and i+1 != j:
              _shift_by = 2
              boubox = _shift_first_last(boubox,_shift_by)
              i -= _shift_by
              j = i+1
              del(_shift_by)

            boubox = numpy.vstack((boubox[:j,:],_pi,boubox[j:,:]))
            del(i,j,_pi)

          del(_iseg,_isegs)
        # 4/4 Regenerate path
          path = mpltPath.Path(boubox)
        del(outside)
      del(inside)

    boubox = numpy.append(boubox, [[nan, nan]], axis=0)

    return inner, mainland, boubox


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


def _clip_polys(polys, bbox, verbose):
    """Clip segments in `polys` that intersect with `bbox`.
    Clipped segments need to extend outside `bbox` to avoid
    false positive `all(inside)` cases. Solution here is to
    add a small offset to the `bbox`.
    """
    if verbose > 1:
        print("Collapsing polygon segments outside bbox...")

    # Inflate bounding box to allow clipped segment to overshoot original box.
    delta = 0.002 # ~2km
    bbox = (bbox[0]-delta,bbox[1]+delta,bbox[2]-delta,bbox[3]+delta)
    boubox = numpy.asarray(_create_boubox(bbox))
    path = mpltPath.Path(boubox)
    polys = _convert_to_list(polys)

    out = []
    for i,poly in enumerate(polys):
      plg = poly[:-2,:]
      inside = path.contains_points(plg)
      if all(inside):
        out.append(poly)
      elif any(inside):
        _ins = numpy.where(inside)[0]
      # Change order of polygon vertices to make sure that the first
      # coordinate falls outside the bounding box.
        if 0 in _ins:
          _shift_by =  numpy.where(~inside)[0][0]
          plg = _shift_first_last(plg,_shift_by)
          inside = path.contains_points(plg)

        _ins = _index_segments_to_list(numpy.where(inside)[0])

        for _seg in _ins:
          _pFi = plg[[_seg[0]-1,_seg[0]],:]
          _pLa = plg[[_seg[-1],_seg[-1]+1],:]
          _pseg = plg[_seg,:]

          _x1,_y1 = _intersect_segment(_pFi,bbox)
          _xL,_yL = _intersect_segment(_pLa,bbox)
          if any(numpy.isnan([_x1,_y1,_xL,_yL])):
            raise ValueError('Could not find intersection between polygon segment and bounding box.')
          if numpy.isclose(_xL-_x1,0.) or numpy.isclose(_yL-_y1,0.):
            _pseg = numpy.vstack((_pseg,[[_xL,_yL],[_x1,_y1]]))
          else:
            _i = mpltPath.Path(poly).contains_points(boubox[:-1,:])
            _iB = numpy.where(_i)[0][0]
            _pseg = numpy.vstack((_pseg,[[_xL,_yL],boubox[_iB,:],[_x1,_y1]]))

          out.append( numpy.vstack((_pseg,_pseg[0,:],[[nan,nan]])) )

    return _convert_to_array(out)


def _nth_simplify(polys, bbox, verbose):
    """Collapse segments in `polys` outside of `bbox`"""
    if verbose > 1:
        print("Collapsing segments outside bbox...")
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
                bd = min(
                    j + 50, len(inside) - 1
                )  # collapses 200 pts to 1 vertex (arbitary)
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


def _index_segments_to_list(s):
    """
    Split indexes that define segments to list of segments.
    """
    _l = []
    _i = numpy.where(numpy.diff(s)>1)[0]+1
    for _a in numpy.split(s,_i):
      _l.append(_a)

    return _l


def _find_closest_index(_p,_s):
    """
    Find closes coordinate array index (_p) based on
    coordinate pair given in _s.
    """
    d = (_p[:,0] - _s[0,0])**2 + (_p[:,1] - _s[0,1])**2
    i = numpy.argmin(d)
    d = (_p[:,0] - _s[-1,0])**2 + (_p[:,1] - _s[-1,1])**2
    j = numpy.argmin(d)

    return i,j


def _is_path_ccw(_p):
    """Compute curve orientation from first two line segment of a polygon.
    Source: https://en.wikipedia.org/wiki/Curve_orientation
    """
    detO = 0.
    O = numpy.ones((3,3))

    i = 0
    while (i+3<_p.shape[0] and numpy.isclose(detO,0.)):
      # Colinear vectors detected. Try again with next 3 indices.
      O[:,1:] = _p[i:i+3,:]
      detO = numpy.linalg.det(O)
      i += 1

    if numpy.isclose(detO,0.):
      raise RuntimeError('Cannot determine orientation from colinear path.')

    return detO>0.


def _shift_first_last(_p,n=2):
    """
    Move first and last point in array to a different array index.
    No NaN in array allowed.
    """
    if n<1 or n>=_p.shape[0]-1:
      raise ValueError('Value for n={:d} out of bounds (0,{:d})'.format(n._p.shape[0]))

    if all(numpy.isclose(_p[0,:],_p[-1,:])):
      _a = _p[:-1,:]
    else:
      _a = _p

    _arr = numpy.vstack((_a[n:,:],_a[0:n,:],_a[n,:]))
    _lenP = _poly_length(_p)
    _lenA = _poly_length(_arr)
    if not numpy.isclose(_lenP,_lenA):
      raise IndexError('Checksum mismatch (unequal arrays circumferences).')

    return _arr

def _intersect_segment(line,bbox):
    """
    Return point intersection of line of 2 points (numpy.array)
    with bounding box (tuple: xmin,xmax,ymin,ymax).
    Note: The solution presented here considers a special case, where
          intersection with horizontal and vertical is found.

    TODO: The general case would consider the solution to 2 linear
          systems using the Cramer's rule. Consider the linear system
            a1*x + b1*y = c1
            a2*x + b2*y = c2
          and assume (a1*b1 - b1*a2) is non-zero. The solution
          (xi,yi) is found with Cramer's rule as
            xi = (c1*b2 - b1*c2) / (a1*b2 - b1*a2)
            yi = (a1*c2 - c1*a2) / (a1*b2 - b1*a2) .
    """
    box = (line[:,0].min(),line[:,0].max(),line[:,1].min(),line[:,1].max())

    if numpy.isclose(line[1,0]-line[0,0], 0.):
    # Special case: intersection of vertical segment.
      xi = line[0,0]
      for yi in bbox[2:]:
        if box[2]<=yi and yi<=box[3]:
          break
        else:
          xi = numpy.nan
          yi = numpy.nan
    else:
    # Convert line segment to linear function y = mx + n
      fit = numpy.zeros(2)
      fit[0] = (line[1,1]-line[0,1]) / (line[1,0]-line[0,0])  # m
      fit[1] = line[0,1] - (line[0,0]*fit[0])                 # n
     #print('line: y = {:.3f}x {:+.3f}'.format(fit[0],fit[1]))

      fit_xi = lambda y,p: (y-p[1]) / p[0]
      fit_yi = lambda x,p: p[0]*x + p[1]

      for i,lim in enumerate(bbox):
        if i<2:
         xi = lim;
         yi = fit_yi(xi,fit)
        elif i<4:
         yi = lim
         xi = fit_xi(yi,fit)

        if box[0]<=xi and xi<=box[1] and box[2]<=yi and yi<=box[3]:
          break
        else:
          xi = nan
          yi = nan

    return xi,yi


# TODO: this should be called "Vector" and contain general methods for vector datasets
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

        if isinstance(bbox,tuple):
          _boubox = numpy.asarray(_create_boubox(bbox)) # polygon is CCW
        else:
          _boubox = numpy.asarray(bbox)
          if not _is_path_ccw(_boubox):
            _boubox = numpy.flipud(_boubox)
          bbox = (numpy.nanmin(_boubox[:,0]),numpy.nanmax(_boubox[:,0]),\
                  numpy.nanmin(_boubox[:,1]),numpy.nanmax(_boubox[:,1]) )

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
        self, ax=None, xlabel=None, ylabel=None, title=None, file_name=None, show=True
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
        (line3,) = ax.plot(self.boubox[:, 0], self.boubox[:, 1], "g-")

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


def _extract_bounds(lons, lats, bbox):
    """Extract the indexes of the subregion"""
    # bounds (from DEM)
    blol, blou = numpy.amin(lons), numpy.amax(lons)
    blal, blau = numpy.amin(lats), numpy.amax(lats)
    # check bounds
    if bbox[0] < blol or bbox[1] > blou:
        raise ValueError(
            "bounding box "
            + str(bbox)
            + " exceeds DEM extents "
            + str((blol, blou, blal, blau))
            + "!"
        )
    if bbox[2] < blal or bbox[3] > blau:
        raise ValueError(
            "bounding box "
            + str(bbox)
            + " exceeds DEM extents "
            + str((blol, blou, blal, blau))
            + "!"
        )
    # latitude lower and upper index
    latli = numpy.argmin(numpy.abs(lats - bbox[2]))
    latui = numpy.argmin(numpy.abs(lats - bbox[3])) + 1
    # longitude lower and upper index
    lonli = numpy.argmin(numpy.abs(lons - bbox[0]))
    lonui = numpy.argmin(numpy.abs(lons - bbox[1])) + 1

    return latli, latui, lonli, lonui


def _from_tif(filename, bbox, verbose):
    """Read in a digitial elevation model from a tif file"""

    if verbose > 0:
        print("Reading in tif... " + filename)

    with Image.open(filename) as img:
        meta_dict = {TAGS[key]: img.tag[key] for key in img.tag.keys()}

    nc, nr = meta_dict["ImageLength"][0], meta_dict["ImageWidth"][0]
    reso = meta_dict["ModelPixelScaleTag"]
    tie = meta_dict["ModelTiepointTag"][3:5]

    lats, lons = [], []
    for n in range(nr):
        lats.extend([tie[1] - reso[1] * n])
    for n in range(nc):
        lons.extend([tie[0] + reso[0] * n])
    lats, lons = numpy.asarray(lats), numpy.asarray(lons)
    # should be increasing
    if lats[1] < lats[0]:
        lats = numpy.flipud(lats)
    if lons[1] < lons[0]:
        lons = numpy.flipud(lons)

    latli, latui, lonli, lonui = _extract_bounds(lons, lats, bbox)

    tmp = numpy.asarray(Image.open(filename))
    topobathy = tmp[latli:latui, lonli:lonui]

    return (lats, slice(latli, latui)), (lons, slice(lonli, lonui)), topobathy


def _from_netcdf(filename, bbox, verbose):
    """Read in digitial elevation model from a NetCDF file"""

    if verbose > 0:
        print("Reading in NetCDF... " + filename)
    # well-known variable names
    wkv_x = ["x", "Longitude", "longitude", "lon"]
    wkv_y = ["y", "Latitude", "latitude", "lat"]
    wkv_z = ["Band1", "z"]
    try:
        with Dataset(filename, "r") as nc_fid:
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
            lons = nc_fid.variables[lon_name][:]
            lats = nc_fid.variables[lat_name][:]
            latli, latui, lonli, lonui = _extract_bounds(lons, lats, bbox)
            topobathy = nc_fid.variables[z_name][latli:latui, lonli:lonui]
            return (lats, slice(latli, latui)), (lons, slice(lonli, lonui)), topobathy
    except IOError:
        print("Unable to open file...quitting")
        quit()


class DEM(Grid):
    """Digitial elevation model read in from a tif or NetCDF file
    parent class is a :class:`Grid`
    """

    def __init__(self, dem, bbox, verbose=1):

        basename, ext = os.path.splitext(dem)
        if ext.lower() in [".nc"]:
            la, lo, topobathy = _from_netcdf(dem, bbox, verbose)
        elif ext.lower() in [".tif"]:
            la, lo, topobathy = _from_tif(dem, bbox, verbose)
        else:
            raise ValueError(
                "DEM file %s has unknown format '%s'." % (self.dem, ext[1:])
            )
        self.dem = dem
        # determine grid spacing in degrees
        lats = la[0]
        lons = lo[0]
        dy = numpy.abs(lats[1] - lats[0])
        dx = numpy.abs(lons[1] - lons[0])
        super().__init__(
            bbox=bbox,
            dx=dx,
            dy=dy,
        )
        self.values = topobathy[: self.ny, : self.nx].T
        super().build_interpolant()
