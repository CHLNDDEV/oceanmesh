import matplotlib.pyplot as plt
import numpy
import scipy.spatial
from scipy.interpolate import RegularGridInterpolator

# controls extrapolation behavior.
_FILL = None  # 999999


def compute_minimum(edge_lengths):
    """Determine the minimum of all edge lengths in the domain"""
    # project all edge_lengths onto the grid of the first one
    base_edge_length = edge_lengths[0]
    edge_lengths = [
        edge_length.project(base_edge_length) for edge_length in edge_lengths[1::]
    ]
    edge_lengths.insert(0, base_edge_length)
    minimum_values = numpy.minimum.reduce(
        [edge_length.values for edge_length in edge_lengths]
    )
    min_edgelength = numpy.amin(minimum_values)
    # construct a new grid object with these values
    grid = Grid(
        bbox=base_edge_length.bbox,
        dx=base_edge_length.dx,
        dy=base_edge_length.dy,
        hmin=min_edgelength,
        values=minimum_values,
    )
    grid.build_interpolant()
    return grid


class Grid:
    """Abstracts a structured grid along with
    primitive operations (e.g., min, project, etc.) and
    stores data `values` defined at each grid point.

    Parameters
    ----------
    bbox: tuple
        domain extents
    dx: float
        spacing between grid points
    values: scalar or array-like
        values at grid points

    Attributes
    ----------
        x0y0: tuple
            bottom left corner coordinate
        nx: int
            number of grid points in x-direction
        ny: int
            number of grid points in y-direction

    """

    def __init__(self, bbox, dx, dy=None, hmin=None, values=None, fill=_FILL):
        if dy is None:
            dy = dx
        self.bbox = bbox
        self.x0y0 = (bbox[0], bbox[2])  # bottom left corner coordinates
        self.dx = dx
        self.dy = dy
        self.hmin = hmin
        self.nx = int((self.bbox[1] - self.bbox[0]) // self.dx) + 1
        self.ny = int((self.bbox[3] - self.bbox[2]) // self.dy) + 1
        self.values = values
        self.eval = None
        self.fill = fill

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        if value < 0:
            raise ValueError("Grid spacing (dx) must be > 0.0")
        self.__dx = value

    @property
    def dy(self):
        return self.__dy

    @dy.setter
    def dy(self, value):
        if value < 0:
            raise ValueError("Grid spacing (dy) must be > 0.0")
        self.__dy = value

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
        elif data is None:
            return
        self.__values = data[: self.nx, : self.ny]

    def create_vectors(self):
        """Build coordinate vectors

        Parameters
        ----------
            None

        Returns
        -------
        x: ndarray
            1D array contain data with `float` type of x-coordinates.
        y: ndarray
            1D array contain data with `float` type of y-coordinates.

        """
        x = self.x0y0[0] + numpy.arange(0, self.nx) * self.dx
        y = self.x0y0[1] + numpy.arange(0, self.ny) * self.dy
        return x, y

    def create_grid(self):
        """Build a structured grid

        Parameters
        ----------
            None

        Returns
        -------
        xg: ndarray
            2D array contain data with `float` type.
        yg: ndarray
            2D array contain data with `float` type.

        """
        x, y = self.create_vectors()
        return numpy.meshgrid(x, y, sparse=False, indexing="ij")

    def find_indices(self, points, lon, lat, tree=None):
        """Find linear indices `indices` into a 2D array such that they
        return the closest point in the structured grid defined by `x` and `y`
        to `points`.

        Parameters
        ----------
        points: ndarray
            Query points. 2D array with `float` type.
        lon: ndarray
            Grid points in x-dimension. 2D array with `float` type.
        lat: ndarray
            Grid points in y-dimension. 2D array with `float` type.
        tree: :obj:`scipy.spatial.ckdtree`, optional
            A KDtree with coordinates from :class:`Shoreline`

        Returns
        -------
        indices: ndarray
            Indicies into an array. 1D array with `int` type.

        """
        points = points[~numpy.isnan(points[:, 0]), :]
        if tree is None:
            lonlat = numpy.column_stack((lon.ravel(), lat.ravel()))
            tree = scipy.spatial.cKDTree(lonlat)
        dist, idx = tree.query(points, k=1)
        return numpy.unravel_index(idx, lon.shape)

    def project(self, grid2):
        """Projects linearly self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.

        Note
        ----
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained. Grid3 has
        dx, dy gridspacing resolution of grid2.

        Parameters
        ----------
        grid2: :obj:`Grid`
            A :obj:`Grid` with `values`.

        Returns
        -------
        grid3: :obj:`Grid`
            A new `obj`:`Grid` with projected `values`.

        """
        # is grid2 even a grid object?
        if not isinstance(grid2, Grid):
            raise ValueError("Both objects must be grids.")
        # check if they overlap
        x1min, x1max, y1min, y1max = self.bbox
        x2min, x2max, y2min, y2max = self.bbox
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        if overlap is False:
            raise ValueError("Grid objects do not overlap.")
        lon1, lat1 = self.create_vectors()
        lon2, lat2 = grid2.create_vectors()
        # take data from grid1 --> grid2
        fp2 = RegularGridInterpolator(
            (lon2, lat2),
            grid2.values,
            method="linear",
            bounds_error=False,
            fill_value=999999,
        )
        # build grid for base
        xg, yg = numpy.meshgrid(lon1, lat1, indexing="ij", sparse=True)
        # interpolate the second onto the first
        new_values = fp2((xg, yg))
        # where the fill value is, replace it with was previously there
        new_values[new_values == 999999] = self.values[new_values == 999999]
        return Grid(
            bbox=self.bbox,
            dx=self.dx,
            dy=self.dy,
            hmin=self.dx,
            values=new_values,
            fill=self.fill,
        )

    def plot(
        self,
        hold=False,
        show=True,
        vmin=None,
        vmax=None,
        coarsen=1,
        xlabel=None,
        ylabel=None,
        title=None,
        cbarlabel=None,
        filename=None,
    ):
        """Visualize the values in :obj:`Grid`

        Parameters
        ----------
        hold: boolean, optional
            Whether to create a new plot axis.

        Returns
        -------
        ax: handle to axis of plot
            handle to axis of plot.

        """

        x, y = self.create_grid()

        fig, ax = plt.subplots()
        ax.axis("equal")
        c = ax.pcolormesh(
            x[:-1:coarsen, :-1:coarsen],
            y[:-1:coarsen, :-1:coarsen],
            self.values[:-1:coarsen, :-1:coarsen],
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        cbar = plt.colorbar(c)
        if cbarlabel is not None:
            cbar.set_label(cbarlabel)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        if hold is False and show:
            plt.show()
        if filename is not None:
            plt.savefig(filename)
        return ax

    def build_interpolant(self):
        """Construct a RegularGriddedInterpolant sizing function stores it as
        the `eval` field.

        Parameters
        ----------
        values: array-like
            An an array of values that form the gridded interpolant:w

        """
        lon1, lat1 = self.create_vectors()

        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=self.fill,
        )

        def sizing_function(x):
            return fp(x)

        self.eval = sizing_function
        return self
