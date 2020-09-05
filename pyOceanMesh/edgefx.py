import numpy
import scipy.spatial
import skfmm

from .geodata import Shoreline

__all__ = ["Grid", "DistanceSizingFunction"]


class Grid:
    def __init__(self, bbox, grid_spacing):
        """Class to create abstract a structured grid"""

        self.x0y0 = (
            min(bbox[0:2]),
            min(bbox[2:]),
        )  # bottom left corner coordinates
        self.grid_spacing = grid_spacing
        ceil, abs = numpy.ceil, numpy.abs
        self.nx = int(ceil(abs(self.x0y0[0] - bbox[1]) / self.grid_spacing))
        self.ny = int(ceil(abs(self.x0y0[1] - bbox[3]) / self.grid_spacing))
        self.bbox = bbox

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

    def create_grid(self):
        x = self.x0y0[0] + numpy.arange(0, self.nx) * self.grid_spacing
        y = self.x0y0[1] + numpy.arange(0, self.ny) * self.grid_spacing
        return numpy.meshgrid(x, y, sparse=False, indexing="ij")

    def find_indices(self, points, lon, lat, tree=None):
        points = points[~numpy.isnan(points[:, 0]), :]
        if tree is None:
            print("Building tree...")
            lonlat = numpy.column_stack((lon.ravel(), lat.ravel()))
            tree = scipy.spatial.cKDTree(lonlat)
        dist, idx = tree.query(points, k=1)
        return numpy.unravel_index(idx, lon.shape)


# TODO overload plus for grid objects


class DistanceSizingFunction(Grid):
    def __init__(self, Shoreline, dis=0.15, max_scale=0.0):
        """Create a sizing function that varies linearly at a rate `dis`
        from the union of the shoreline points"""
        print("Building distance function...")
        self.Shoreline = Shoreline
        super().__init__(bbox=Shoreline.bbox, grid_spacing=Shoreline.h0)
        # create phi (1 where shoreline point intersects grid 0 elsewhere)
        phi = numpy.ones(shape=(self.nx, self.ny))
        lon, lat = self.create_grid()
        points = numpy.vstack((Shoreline.inner, Shoreline.mainland))
        indices = self.find_indices(points, lon, lat)
        phi[indices] = -1.0

        self.DistanceSizing = skfmm.distance(phi, self.grid_spacing, narrow=max_scale)

    @property
    def Shoreline(self):
        return self.__Shoreline

    @Shoreline.setter
    def Shoreline(self, obj):
        if not isinstance(obj, Shoreline):
            raise ValueError
        self.__Shoreline = obj

    @property
    def DistanceSizing(self):
        return self.__DistanceSizing

    @DistanceSizing.setter
    def DistanceSizing(self, vals):
        self.__DistanceSizing = vals

    def plot(self, hold=False):
        """Visualize the distance function"""
        import matplotlib.pyplot as plt

        xmin, xmax, ymin, ymax = self.bbox
        x = numpy.arange(xmin, xmax, self.grid_spacing * 10)
        y = numpy.arange(ymin, ymax, self.grid_spacing * 10)

        fig, ax = plt.subplots()
        cs = ax.pcolorfast(x, y, self.DistanceSizing.T,vmin=0.0, vmax=0.10)
        plt.title("Distance Sizing Function")
        ax.axis("equal")
        if hold is False:
            plt.show()
        return ax
