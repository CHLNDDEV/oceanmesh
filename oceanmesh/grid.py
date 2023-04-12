import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from scipy.interpolate import RegularGridInterpolator

from .idw import Invdisttree
from .region import Region

logger = logging.getLogger(__name__)


def compute_minimum(edge_lengths):
    """Determine the minimum of all edge lengths in the domain"""
    _crs = edge_lengths[0].crs
    msg = "All edgelengths must have the same CRS"
    for el in edge_lengths[1::]:
        assert _crs == el.crs, msg
    # project all edge_lengths onto the grid of the first one
    base_edge_length = edge_lengths[0]
    edge_lengths = [
        edge_length.interpolate_to(base_edge_length)
        for edge_length in edge_lengths[1::]
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
        extrapolate=True,
        crs=base_edge_length.crs,
    )

    grid.build_interpolant()
    return grid


class Grid(Region):
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
    crs: pyproj.PROJ, optional
        Well-known text (WKT)
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

    def __init__(
        self,
        bbox,
        dx,
        dy=None,
        crs="EPSG:4326",
        hmin=None,
        values=None,
        extrapolate=False,
    ):
        super().__init__(bbox, crs)
        if dy is None:
            dy = dx # equidistant grid in both x and y dirs if not passed
        self.bbox = bbox
        self.x0y0 = (bbox[0], bbox[2])  # bottom left corner coordinates (x,y)
        self.dx = dx
        self.dy = dy
        self.nx = None  # int((self.bbox[1] - self.bbox[0]) // self.dx) + 1
        self.ny = None  # int((self.bbox[3] - self.bbox[2]) // self.dy) + 1
        self.values = values
        self.eval = None
        self.extrapolate = extrapolate
        self.hmin = hmin

    @property
    def dx(self):
        return self.__dx

    @dx.setter
    def dx(self, value):
        if value <= 0:
            raise ValueError("Grid spacing (dx) must be >= 0.0")
        self.__dx = value

    @property
    def dy(self):
        return self.__dy

    @dy.setter
    def dy(self, value):
        if value <= 0:
            raise ValueError("Grid spacing (dy) must be >= 0.0")
        self.__dy = value

    @property
    def values(self):
        return self.__values

    @values.setter
    def values(self, data):
        if np.isscalar(data):
            self.nx = int((self.bbox[1] - self.bbox[0]) // self.dx) + 1
            self.ny = abs(int((self.bbox[3] - self.bbox[2]) // self.dy) + 1)
            data = np.tile(data, (self.nx, self.ny))
        self.__values = data
        self.nx, self.ny = data.shape

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
        x = self.x0y0[0] + np.arange(0, self.nx) * self.dx  # ascending monotonically
        y = self.x0y0[1] + np.arange(0, self.ny) * abs(self.dy)             
        y  = y[::-1] # descending monotonically
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
        xg, yg = np.meshgrid(x, y, indexing="ij")
        return xg, yg

    def find_indices(self, points, lon, lat, tree=None, k=1):
        """Find linear indices `indices` into a 2D array such that they
        return the closest k point(s) in the structured grid defined by `x` and `y`
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
        k: int, optional
            Number of closest points to return

        Returns
        -------
        indices: ndarray
            Indicies into an array. 1D array with `int` type.

        """
        points = points[~np.isnan(points[:, 0]), :]
        if tree is None:
           lonlat = np.column_stack((lon.ravel(), lat.ravel()))
           tree = scipy.spatial.cKDTree(lonlat)
        try:
           dist, idx = tree.query(points, k=k, workers=-1)
        except (Exception,):
           dist, idx = tree.query(points, k=k, n_jobs=-1)
        return np.unravel_index(idx, lon.shape)

    def interpolate_to(self, grid2, method="nearest"):
        """Interpolates self.values onto :class`Grid` grid2 forming a new
        :class:`Grid` object grid3.
        Note
        ----
        In other words, in areas of overlap, grid1 values
        take precedence elsewhere grid2 values are retained. Grid3 has
        dx & dy grid spacing following the resolution of grid2.
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
            hmin=grid2.hmin,
            values=new_values,
            crs=grid2.crs,
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
        x1min, x1max, y1min, y1max = coarse.bbox
        x2min, x2max, y2min, y2max = self.bbox
        overlap = (x1min < x2min) & (x1max > x2max) & (y1min < y2min) & (y1max > y2max)
        if not overlap:
            logger.warning("Grid objects do not overlap.")
            return coarse
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
        holding=False,
        coarsen=1,
        plot_colorbar=False,
        **kwargs,
    ):
        """Visualize the values in :obj:`Grid`

        Parameters
        ----------
        holding: boolean, optional
            Whether to create a new plot axis.

        Returns
        -------
        fig:
        ax: handle to axis of plot
            handle to axis of plot.

        """
        _xg,_yg=self.create_grid()
        fig, ax = plt.subplots()
        ax.axis("equal")
        pc = ax.pcolor(
            _xg[::coarsen,::coarsen],
            _yg[::coarsen,::coarsen],
            self.values[::coarsen, ::coarsen],
            **kwargs,
        )
        if plot_colorbar:
            fig.colorbar(pc)
        if holding is False:
            plt.show()
        return fig, ax, pc

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
