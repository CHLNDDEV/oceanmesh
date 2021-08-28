import warnings

import numpy
import skfmm

from _HamiltonJacobi import gradient_limit
from .grid import Grid

__all__ = [
    "enforce_mesh_gradation",
    "enforce_mesh_size_bounds_elevation",
    "distance_sizing_function",
    "wavelength_sizing_function",
    "create_multiscale",
]


def enforce_mesh_size_bounds_elevation(grid, dem, bounds, verbose=1):
    """Enforce mesh size bounds as a function of elevation

    Parameters
    ----------
    grid: :class:`Grid`
        A grid object with its values field populated
    dem:  :class:`Dem`
        Data processed from :class:`Dem`.
    bounds: list of list
        A list of potentially > 1 len(4) lists containing
        [[min_mesh_size, max_mesh_size, min_elevation_bound, max_elevation_bound]]
        The orientation of the elevation bounds should be the same as that of the DEM
        (i.e., negative downwards towards the Earth's center).
    verbose: boolean
        Whether or not to print messages to the screen

    Returns
    -------
    :class:`Grid` object
        A sizing function with the bounds mesh size bounds enforced.
    """
    lon, lat = grid.create_grid()
    tmpz = dem.eval((lon, lat))
    for i, bound in enumerate(bounds):
        assert (
            len(bound) == 4
        ), "Bounds must be specified  as a list with [min_mesh_size, max_mesh_size, min_elevation_bound, max_elevation_bound]"
        min_h, max_h, min_z, max_z = bound
        # for now do this crude conversion
        min_h /= 111e3
        max_h /= 111e3
        # sanity checks
        error_sz = f"For bound number {i} the maximum size bound {max_h} is smaller than the minimum size bound {min_h}"
        error_elev = f"For bound number {i} the maximum elevation bound {max_z} is smaller than the minimum elevation bound {min_z}"
        assert min_h < max_h, error_sz
        assert min_z < max_z, error_elev
        # get grid values to enforce the bounds
        upper_indices = numpy.where(
            (tmpz > min_z) & (tmpz <= max_z) & (grid.values >= max_h)
        )
        lower_indices = numpy.where(
            (tmpz > min_z) & (tmpz <= max_z) & (grid.values < min_h)
        )

        grid.values[upper_indices] = max_h
        grid.values[lower_indices] = min_h

    grid.build_interpolant()

    return grid


def enforce_mesh_gradation(grid, gradation=0.15, verbose=1):
    """Enforce a mesh size gradation bound `gradation` on a :class:`grid`

    Parameters
    ----------
    grid: :class:`Grid`
        A grid object with its values field populated
    gradation: float
        The decimal percent mesh size gradation rate to-be-enforced.
    verbose: boolean
        whether to write messages to the screen

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field gradient limited

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


def create_multiscale(list_of_grids, verbose=True):
    """Given a list of mesh size functions in a hiearcharcy
    w.r.t. to minimum mesh size (largest -> smallest),
    create a so-called multiscale mesh size function

    Parameters
    ----------
    list_of_grids: a Python list
        A list containing grids with resolution in decreasing order.
    verbose: boolean, optional
        Whether to write messages to the screen

    Returns
    -------
    new_list_of_grids: a Python list
        Same as input but with relatively finer grids projected and graded onto the relatively coarser ones.
    minimum_edge_length: float
        The minimum edge length in meters throughout the domain

    """

    new_list_of_grids = []
    # loop through remaining sizing functions
    for idx1, new_coarse in enumerate(list_of_grids[:-1]):
        # project all finer onto coarse and enforce gradation
        print(f"For sizing function #{idx1}")
        for k, finer in enumerate(list_of_grids[idx1 + 1 :]):
            if verbose:
                print(
                    f"  Projecting sizing function #{idx1+1 + k} onto sizing function #{idx1}"
                )
            new_coarse = finer.project(new_coarse)
            # enforce mesh size gradation
            new_coarse = enforce_mesh_gradation(new_coarse, verbose=0)
        new_list_of_grids.append(new_coarse)
    # retain the finest
    new_list_of_grids.append(list_of_grids[-1])

    # debug
    k = 0
    for a, b in zip(list_of_grids, new_list_of_grids):
        a.plot(show=False, filename=f"org{k}.png")
        b.plot(show=False, filename=f"new{k}.png")
        k += 1

    # compute new minimum edge length to mesh with
    minimum_edge_length = 99999
    for func in new_list_of_grids:
        minimum_edge_length = numpy.amin([func.dx, func.dy, minimum_edge_length])

    # return the mesh size function to query during genertaion
    def wrapper(qpts):
        hmin = numpy.array([len(qpts)] * 9999)
        for func in new_list_of_grids:
            h = func.eval(qpts)
            hmin = numpy.minimum(h, hmin)

    return wrapper, minimum_edge_length
