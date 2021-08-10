import warnings

import numpy
import skfmm

from .cpp.HamiltonJacobi import gradient_limit
from .grid import Grid

__all__ = [
    "enforce_mesh_gradation",
    "distance_sizing_function",
    "wavelength_sizing_function",
]


def enforce_mesh_gradation(grid, gradation=0.15, verbose=1):
    """Enforce a mesh size gradation bound `gradation` on a :class:`grid`

    Parameters
    ---------

    Returns
    -------

    """
    if gradation < 0:
        raise ValueError("Parameter `gradation` must be > 0.0")
    if gradation > 1.0:
        warnings.warn("Parameter `gradation` is set excessively high (> 1.0)")
    if verbose:
        print(f"Enforcing mesh size gradation of {gradation} decimal percent...")

    elen = grid.dx
    if grid.dx != grid.dy:
        assert "Structured grids with unequal grid spaces not yet supported"
    cell_size = grid.values.copy()
    sz = cell_size.shape
    sz = (sz[0], sz[1], 1)
    cell_size = cell_size.flatten("F")
    tmp = gradient_limit([*sz], elen, gradation, 10000, cell_size)
    tmp = numpy.reshape(tmp, (sz[0], sz[1]), "F")
    grid_limited = Grid(bbox=grid.bbox, dx=grid.dx, values=tmp, hmin=grid.hmin)
    grid_limited.build_interpolant()
    return grid_limited


def distance_sizing_function(
    shoreline, rate=0.15, max_edge_length=None, verbose=1, coarsen=1
):
    """Mesh sizes that vary linearly at `rate` from coordinates in `obj`:Shoreline

    Parameters
    ----------
    shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    rate: float, optional
        The rate of expansion in decimal percent from the shoreline.
    max_edge_length: float, optional
        The maximum allowable edge length
    verbose: boolean, optional
        Whether to write messages to the screen
    coarsen: integer, optional
        Downsample the grid by a constant factor in x and y axes

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    if verbose > 0:
        print("Building a distance sizing function...")
    grid = Grid(bbox=shoreline.bbox, dx=shoreline.h0 * coarsen, hmin=shoreline.h0)
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = numpy.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = numpy.vstack((shoreline.inner, shoreline.mainland))
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    dis = numpy.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
    grid.values = shoreline.h0 + dis * rate

    if max_edge_length is not None:
        max_edge_length /= 111e3  # assume the value is passed in meters
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.build_interpolant()
    return grid


def wavelength_sizing_function(
    dem,
    wl=10,
    min_edgelength=None,
    max_edge_length=None,
    verbose=1,
):
    """Mesh sizes that vary proportional to an estimate of the wavelength
       of the M2 tidal constituent

    Parameters
    ----------
    dem:  :class:`Dem`
        Data processed from :class:`Dem`.
    wl: integer, optional
        The number of desired elements per wavelength of the M2 constituent
    min_edgelength: float, optional
        The minimum edge length in meters in the domain. If None, the min
        of the edgelength function is used.
    max_edge_length: float, optional
        The maximum edge length in meters in the domain.
    verbose: boolean, optional
        Whether to write messages to the screen


    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    if verbose > 0:
        print("Building a wavelength sizing function...")
    lon, lat = dem.create_grid()
    tmpz = dem.eval((lon, lat))

    grav = 9.807
    period = 12.42 * 3600  # M2 period in seconds
    grid = Grid(bbox=dem.bbox, dx=dem.dx, dy=dem.dy)
    tmpz[numpy.abs(tmpz) < 1] = 1
    grid.values = period * numpy.sqrt(grav * numpy.abs(tmpz)) / wl
    grid.values /= 111e3  # transform to degrees
    if min_edgelength is None:
        min_edgelength = numpy.amin(grid.values)
    grid.hmin = min_edgelength
    if max_edge_length is not None:
        max_edge_length /= 111e3
        grid.values[grid.values > max_edge_length] = max_edge_length
    grid.build_interpolant()
    return grid
