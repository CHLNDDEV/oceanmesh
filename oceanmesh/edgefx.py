import numpy
import skfmm

from .grid import Grid

__all__ = ["distance_sizing_function", "wavelength_sizing_function"]


def distance_sizing_function(shoreline, rate=0.15, max_edgelength=None):
    """Mesh sizes that vary linearly at `rate` from coordinates in `obj`:Shoreline

    Parameters
    ----------
    shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    rate: float, optional
        The rate of expansion in decimal percent from the shoreline.

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    if max_edgelength is not None:
        max_edgelength /= 111e3  # assume the value is passed in meters
    grid = Grid(bbox=shoreline.bbox, grid_spacing=shoreline.h0, hmin=shoreline.h0)
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = numpy.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = numpy.vstack((shoreline.inner, shoreline.mainland))
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    dis = numpy.abs(skfmm.distance(phi, grid.grid_spacing))
    grid.values = shoreline.h0 + dis * rate
    grid.values[grid.values > max_edgelength] = max_edgelength
    grid.build_interpolant()
    return grid


def wavelength_sizing_function(dem, wl=10, min_edgelength=None, max_edgelength=None):
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
    max_edgelength: float, optional
        The maximum edge length in meters in the domain.

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    xg, yg = dem.get_grid()
    tmpz = dem.Fb((xg, yg))

    grav = 9.807
    period = 12.42 * 3600  # M2 period in seconds
    grid = Grid(bbox=dem.bbox, grid_spacing=dem.grid_spacing)
    grid.values = period * numpy.sqrt(grav * max(abs(tmpz), 1)) / wl
    if min_edgelength is not None:
        min_edgelength = numpy.amin(grid.values)
    grid.hmin = min_edgelength
    if max_edgelength is not None:
        max_edgelength /= 111e3
        grid.values[grid.values > max_edgelength] = max_edgelength
    return grid
