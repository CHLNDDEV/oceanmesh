import numpy
import scipy.spatial
import skfmm
from scipy.interpolate import RegularGridInterpolator

from .geodata import Shoreline

__all__ = ["Grid", "DistanceSizingFunction"]

fill = -99999.0


class Grid:
    def __init__(self, bbox, grid_spacing, values=None):
        """Class to abstract a structured grid and operations define on it
        with data `values` defined at each grid point.
        """

        self.x0y0 = (
            min(bbox[0:2]),
            min(bbox[2:]),
        )  # bottom left corner coordinates
        self.grid_spacing = grid_spacing
        ceil, abs = numpy.ceil, numpy.abs
        self.nx = int(ceil(abs(self.x0y0[0] - bbox[1]) / self.grid_spacing)) + 1
        self.ny = int(ceil(abs(self.x0y0[1] - bbox[3]) / self.grid_spacing)) + 1
        self.bbox = bbox
        self.values = values

    @property
    def grid_spacing(self):
        return self.__grid_spacing

    @grid_spacing.setter
    def grid_spacing(self, value):
        if value < 0:
            raise ValueError("Grid spacing must be > 0.0")
        self.__grid_spacing = value

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
    def values(self):
        return self.__values

    @values.setter
    def values(self, data):
        if numpy.isscalar(data):
            data = numpy.tile(data, (self.nx, self.ny))
        if data.shape != (self.nx, self.ny):
            print("Shape of values does not match grid size")
            raise ValueError
        self.__values = data

    def create_vecs(self):
        x = self.x0y0[0] + numpy.arange(0, self.nx) * self.grid_spacing
        y = self.x0y0[1] + numpy.arange(0, self.ny) * self.grid_spacing
        return x, y

    def create_grid(self):
        x, y = self.create_vecs()
        return numpy.meshgrid(x, y, sparse=False, indexing="ij")

    def find_indices(self, points, lon, lat, tree=None):
        points = points[~numpy.isnan(points[:, 0]), :]
        if tree is None:
            print("Building tree...")
            lonlat = numpy.column_stack((lon.ravel(), lat.ravel()))
            tree = scipy.spatial.cKDTree(lonlat)
        dist, idx = self.tree.query(points, k=1)
        return numpy.unravel_index(idx, lon.shape)

    def project(self, grid2):
        """Projects linearly self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained. Grid3 has
        grid_spacing and resolution of grid2."""
        # is grid2 even a grid object?
        if not isinstance(grid2, Grid):
            print("Both objects must be grids")
            raise ValueError
        # check if they overlap
        x1min, x1max, y1min, y1max = self.bbox
        x2min, x2max, y2min, y2max = self.bbox
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        if overlap is False:
            print("Grid objects do not overlap, nothing to do.")
            raise ValueError
        lon1, lat1 = self.create_vecs()
        lon2, lat2 = grid2.create_vecs()
        # take data from grid1 --> grid2
        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=fill,
        )
        xg, yg = numpy.meshgrid(lon2, lat2, indexing="ij", sparse=True)
        new_values = fp((xg, yg))
        # where fill replace with grid2 values
        new_values[new_values == fill] = grid2.values[new_values == fill]
        return Grid(bbox=grid2.bbox, grid_spacing=grid2.grid_spacing, values=new_values)

    def plot(self, hold=False):
        """Visualize the values in :class:`Grid`"""
        import matplotlib.pyplot as plt

        x, y = self.create_vecs()

        fig, ax = plt.subplots()
        ax.pcolorfast(x, y, self.values)
        ax.axis("equal")
        if hold is False:
            plt.show()
        return ax


class DistanceSizingFunction(Grid):
    def __init__(self, Shoreline, rate=0.15, max_scale=0.0):
        """Create a sizing function that varies linearly at a rate `dis`
        from the union of the shoreline points"""
        print("Building distance function...")
        self.Shoreline = Shoreline
        super().__init__(bbox=Shoreline.bbox, grid_spacing=Shoreline.h0)
        # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
        phi = numpy.ones(shape=(self.nx, self.ny))
        lon, lat = self.create_grid()
        points = numpy.vstack((Shoreline.inner, Shoreline.mainland))
        indices = self.find_indices(points, lon, lat)
        phi[indices] = -1.0
        # call Fast Marching Method
        dis = skfmm.distance(phi, self.grid_spacing, narrow=max_scale)
        self.values = Shoreline.h0 + dis.T * rate

    @property
    def Shoreline(self):
        return self.__Shoreline

    @Shoreline.setter
    def Shoreline(self, obj):
        if not isinstance(obj, Shoreline):
            raise ValueError
        self.__Shoreline = obj
