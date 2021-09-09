import warnings

import numpy
import skfmm
from _HamiltonJacobi import gradient_limit
from inpoly import inpoly2

from . import edges
from .grid import Grid

__all__ = [
    "enforce_mesh_gradation",
    "enforce_mesh_size_bounds_elevation",
    "distance_sizing_function",
    "wavelength_sizing_function",
    "multiscale_sizing_function",
]


def enforce_mesh_size_bounds_elevation(grid, dem, bounds, verbose=True):
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
        assert len(bound) == 4, (
            "Bounds must be specified  as a list with [min_mesh_size,"
            " max_mesh_size, min_elevation_bound, max_elevation_bound]"
        )
        min_h, max_h, min_z, max_z = bound
        # for now do this crude conversion
        min_h /= 111e3
        max_h /= 111e3
        # sanity checks
        error_sz = (
            f"For bound number {i} the maximum size bound {max_h} is smaller"
            f" than the minimum size bound {min_h}"
        )
        error_elev = (
            f"For bound number {i} the maximum elevation bound {max_z} is"
            f" smaller than the minimum elevation bound {min_z}"
        )
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


def enforce_mesh_gradation(grid, gradation=0.15, verbose=True):
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
    grid_limited = Grid(
        bbox=grid.bbox,
        dx=grid.dx,
        values=tmp,
        hmin=grid.hmin,
        extrapolate=grid.extrapolate,
    )
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
    grid = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0 * coarsen,
        hmin=shoreline.h0,
        extrapolate=True,
    )
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = numpy.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = numpy.vstack((shoreline.inner, shoreline.mainland))
    # remove shoreline components outside the shoreline.boubox
    boubox = shoreline.boubox
    e_box = edges.get_poly_edges(boubox)
    mask = numpy.ones((grid.nx, grid.ny), dtype=bool)
    if len(points) > 0:
        try:
            in_boubox, _ = inpoly2(points, boubox, e_box)
            points = points[in_boubox]

            qpts = numpy.column_stack((lon.flatten(), lat.flatten()))
            in_boubox, _ = inpoly2(qpts, shoreline.boubox, e_box)
            mask_indices = grid.find_indices(qpts[in_boubox, :], lon, lat)
            mask[mask_indices] = False
        except (Exception,):
            ...

    # find location of points on grid
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    dis = numpy.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
    tmp = shoreline.h0 + dis * rate
    if max_edge_length is not None:
        max_edge_length /= 111e3  # assume the value is passed in meters
        tmp[tmp > max_edge_length] = max_edge_length

    grid.values = numpy.ma.array(tmp, mask=mask)
    grid.build_interpolant()
    return grid


def wavelength_sizing_function(
    dem,
    wl=10,
    min_edgelength=None,
    max_edge_length=None,
    verbose=True,
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
    grid = Grid(bbox=dem.bbox, dx=dem.dx, dy=dem.dy, extrapolate=True)
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


def multiscale_sizing_function(
    list_of_grids, p=3, nnear=28, blend_width=1000, verbose=True
):
    """Given a list of mesh size functions in a hierarchy
    w.r.t. to minimum mesh size (largest -> smallest),
    create a so-called multiscale mesh size function

    Parameters
    ----------
    list_of_grids: a Python list
        A list containing grids with resolution in decreasing order.
    p: int, optional
        use 1 / distance**p weights nearer points more, farther points less.
    nnear: int, optional
        how many nearest neighbors should one take to perform IDW interp?
    blend_width: float, optional
        The width of the blending zone between nests in meters
    verbose: boolean, optional
        Whether to write messages to the screen

    Returns
    -------
    func: a function
        The global sizing funcion defined over the union of all domains
    new_list_of_grids: a list of  function
        A list of sizing function that takes a point and returns a value

    """
    err = "grid objects must appear in order of descending dx spacing"
    for i, grid in enumerate(list_of_grids[:-1]):
        assert grid.dx >= list_of_grids[i + 1].dx, err

    new_list_of_grids = []
    # loop through remaining sizing functions
    for idx1, new_coarse in enumerate(list_of_grids[:-1]):
        if verbose:
            print(f"For sizing function #{idx1}")
        # interpolate all finer nests onto coarse func and enforce gradation rate
        for k, finer in enumerate(list_of_grids[idx1 + 1 :]):
            if verbose:
                print(
                    f"  Interpolating sizing function #{idx1+1 + k} onto sizing"
                    f" function #{idx1}"
                )
            _dx = finer.dx * 111e3
            _blend_width = int(numpy.floor(blend_width / _dx))
            finer.extrapolate = False
            new_coarse = finer.blend_into(
                new_coarse, blend_width=_blend_width, p=p, nnear=nnear
            )
            new_coarse.extrapolate = True

        # recreate the interpolant
        new_coarse.build_interpolant()
        # append it to list
        new_list_of_grids.append(new_coarse)

    list_of_grids[-1].extrapolate = True
    list_of_grids[-1].build_interpolant()

    # retain the finest
    new_list_of_grids.append(list_of_grids[-1])

    # return the mesh size function to query during genertaion
    # NB: only keep the minimum value over all grids
    def func(qpts):
        hmin = numpy.array([999999] * len(qpts))
        for i, grid in enumerate(new_list_of_grids):
            if i == 0:
                grid.extrapolate = True
            else:
                grid.extrapolate = False
            grid.build_interpolant()
            _hmin = grid.eval(qpts)
            hmin = numpy.min(numpy.column_stack([_hmin, hmin]), axis=1)
        return hmin

    return func, new_list_of_grids
