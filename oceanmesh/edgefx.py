import numpy
import skfmm

from .grid import Grid

__all__ = ["distance_sizing_function"]


def distance_sizing_function(shoreline, rate=0.15, max_scale=0.0):
    """Mesh sizes that vary linearly at `rate` from coordinates in `obj`:Shoreline

    Parameters
    ----------
    Shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    rate: float, optional
        The rate of expansion in decimal percent from the shoreline.
    max_scale: float, optional
        Distance is only calculated in narrow band of `max_scale` width.

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """

    print("Building distance function...")
    grid = Grid(bbox=shoreline.bbox, grid_spacing=shoreline.h0)
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = numpy.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = numpy.vstack((shoreline.inner, shoreline.mainland))
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    # call Fast Marching Method
    dis = skfmm.distance(phi, grid.grid_spacing, narrow=max_scale)
    grid.values = shoreline.h0 + dis * rate
    return grid
