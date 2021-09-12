import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.interpolate import RegularGridInterpolator

from .idw import Invdisttree


def compute_minimum(edge_lengths):
    """Determine the minimum of all edge lengths in the domain"""
    # project all edge_lengths onto the grid of the first one
    base_edge_length = edge_lengths[0]
    edge_lengths = [
        edge_length.project(base_edge_length) for edge_length in edge_lengths[1::]
    ]
    edge_lengths.insert(0, base_edge_length)
    minimum_values = np.minimum.reduce(
        [edge_length.values for edge_length in edge_lengths]
    )
    min_edgelength = np.amin(minimum_values)
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
        domain extent
    dx: float
        spacing between grid points along x-axis
    dy: float, optional
        spacing grid grid points along y-axis
    hmin: float, optional
        minimum grid spacing in domain
    values: scalar or array-like
        values at grid points. If scalar, then an array of
        the value is created matching the extent.
    extrapolate: boolean, optional
        Whether the grid can extrapolate outside its bbox

    Attributes
    ----------
        x0y0: tuple
            bottom left corner coordinate
        nx: int
            number of grid points in x-direction
        ny: int
            number of grid points in y-direction
        eval: func
            A function that takes a vector of points
            and returns a vector of values
    """

    def __init__(self, bbox, dx, dy=None, hmin=None, values=None, extrapolate=False):
        if dy is None:
            dy = dx
        self.bbox = bbox
        self.x0y0 = (bbox[0], bbox[2])  # bottom left corner coordinates
        self.dx = dx
        self.dy = dy
        self.nx = int((self.bbox[1] - self.bbox[0]) // self.dx) + 1
        self.ny = int((self.bbox[3] - self.bbox[2]) // self.dy) + 1
        self.values = values
        self.eval = None
        self.extrapolate = extrapolate
        self.hmin = hmin

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
        if np.isscalar(data):
            data = np.tile(data, (self.nx, self.ny))
        elif data is None:
            return
        self.__values = data[: self.nx, : self.ny]

    @staticmethod
    def get_border(self, arr):
        """Get the border values of a 2D array"""
        return np.concatenate(
            [arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]], axis=0
        )

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
        x = self.x0y0[0] + np.arange(0, self.nx) * self.dx
        y = self.x0y0[1] + np.arange(0, self.ny) * self.dy
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
        return np.meshgrid(x, y, sparse=False, indexing="ij")

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
        points = points[~np.isnan(points[:, 0]), :]
        if tree is None:
            lonlat = np.column_stack((lon.ravel(), lat.ravel()))
            tree = scipy.spatial.cKDTree(lonlat)
        dist, idx = tree.query(points, k=1, workers=-1)
        return np.unravel_index(idx, lon.shape)

    def interpolate_to(self, grid2, method="linear"):
        """Interpolates linearly self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.
        Note
        ----
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained. Grid3 has
        dx and resolution of grid2.
        Parameters
        ----------
        grid2: :obj:`Grid`
            A :obj:`Grid` with `values`.
        method: str, optional
            Way to interpolate data between grids
        Returns
        -------
        grid3: :obj:`Grid`
            A new `obj`:`Grid` with projected `values`.
        """
        # is grid2 even a grid object?
        assert isinstance(grid2, Grid), "Object must be Grid."
        # check if they overlap
        x1min, x1max, y1min, y1max = self.bbox
        x2min, x2max, y2min, y2max = self.bbox
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        assert overlap, "Grid objects do not overlap."
        lon1, lat1 = self.create_vectors()
        lon2, lat2 = grid2.create_vectors()
        if self.extrapolate:
            _FILL = None
        else:
            _FILL = -999
        # take data from grid1 --> grid2
        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method=method,
            bounds_error=False,
            fill_value=_FILL,
        )
        xg, yg = np.meshgrid(lon2, lat2, indexing="ij", sparse=True)
        new_values = fp((xg, yg))
        # where fill replace with grid2 values
        new_values[new_values == _FILL] = grid2.values[new_values == _FILL]
        return Grid(
            bbox=grid2.bbox,
            dx=grid2.dx,
            dy=grid2.dy,
            values=new_values,
        )

    def blend_into(self, coarse, blend_width=10, p=1, nnear=6, eps=0.0):
        """Blend self.values into the values of the coarse grid one so that the
           values transition smoothly. The kwargs control the blending procedure.
        Parameters
        ----------
        coarse: :class:`Grid`
        blend_width: int, optional
            The width of the padding in number of grid points
        p: int, optional
            The polynomial order in the distance weighting scheme
        nnear: int, optional
            The number of nearest points to use to interpolate each point
        eps: float, optional
            Points less than `eps` are considered the same point
        Returns
        -------
        _coarse_w_fine: :class:`Grid`
            The coarse grid with the finer grid interpolated and blended.
        """
        _FILL = -99999  # uncommon value
        if not isinstance(coarse, Grid):
            raise ValueError("Object must be Grid.")
        # check if they overlap
        x1min, x1max, y1min, y1max = self.bbox
        x2min, x2max, y2min, y2max = self.bbox
        overlap = x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max
        assert overlap, "Grid objects do not overlap."
        _fine = self.values
        # 1. Pad the finer grid's values
        _fine_w_pad_values = np.pad(
            _fine, pad_width=blend_width, mode="constant", constant_values=_FILL
        )
        # 2. Create a new Grid fine_w_pad
        _add_length = self.dx * blend_width
        _add_height = self.dy * blend_width
        _new_fine_bbox = (
            self.bbox[0] - _add_length,
            self.bbox[1] + _add_length,
            self.bbox[2] - _add_height,
            self.bbox[3] + _add_height,
        )
        _fine_w_pad = Grid(
            _new_fine_bbox,
            self.dx,
            dy=self.dy,
            values=_fine_w_pad_values,
            extrapolate=self.extrapolate,
        )
        # 2. Interpolate _fine_w_pad onto coarse
        _coarse_w_fine = _fine_w_pad.interpolate_to(coarse, method="nearest")

        # 3. Perform inverse distance weighting on the points with 0
        _xg, _yg = _coarse_w_fine.create_grid()
        _pts = np.column_stack((_xg.flatten(), _yg.flatten()))
        _vals = _coarse_w_fine.values.flatten()

        # find buffer
        ask_index = _vals == _FILL
        known_index = _vals != _FILL

        _tree = Invdisttree(_pts[known_index], _vals[known_index])
        _vals[ask_index] = _tree(_pts[ask_index], nnear=nnear, eps=eps, p=p)

        _hmin = np.amin(_vals[ask_index])
        _coarse_w_fine.hmin = _hmin
        # put it back
        _coarse_w_fine.values = _vals.reshape(*_coarse_w_fine.values.shape)

        return _coarse_w_fine

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
        if self.extrapolate:
            _FILL = None
        else:
            _FILL = 999999

        fp = RegularGridInterpolator(
            (lon1, lat1),
            self.values,
            method="linear",
            bounds_error=False,
            fill_value=_FILL,
        )

        def sizing_function(x):
            return fp(x)

        self.eval = sizing_function
        return self
