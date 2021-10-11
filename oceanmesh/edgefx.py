import logging

import numpy as np
import scipy.spatial
import skfmm
from _HamiltonJacobi import gradient_limit
from inpoly import inpoly2

from . import edges
from .grid import Grid

logger = logging.getLogger(__name__)

__all__ = [
    "enforce_mesh_gradation",
    "enforce_mesh_size_bounds_elevation",
    "distance_sizing_function",
    "wavelength_sizing_function",
    "multiscale_sizing_function",
    "feature_sizing_function",
    "slope_sizing_function",
]


def enforce_mesh_size_bounds_elevation(grid, dem, bounds):
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
        upper_indices = np.where(
            (tmpz > min_z) & (tmpz <= max_z) & (grid.values >= max_h)
        )
        lower_indices = np.where(
            (tmpz > min_z) & (tmpz <= max_z) & (grid.values < min_h)
        )

        grid.values[upper_indices] = max_h
        grid.values[lower_indices] = min_h

    grid.build_interpolant()

    return grid


def enforce_mesh_gradation(grid, gradation=0.15, crs=4326):
    """Enforce a mesh size gradation bound `gradation` on a :class:`grid`

    Parameters
    ----------
    grid: :class:`Grid`
        A grid object with its values field populated
    gradation: float
        The decimal percent mesh size gradation rate to-be-enforced.

    Returns
    -------
    grid: class:`Grid`
        A grid ojbect with its values field gradient limited

    """
    if gradation < 0:
        raise ValueError("Parameter `gradation` must be > 0.0")
    if gradation > 1.0:
        logger.warning("Parameter `gradation` is set excessively high (> 1.0)")

    logger.info(f"Enforcing mesh size gradation of {gradation} decimal percent...")

    elen = grid.dx

    if grid.dx != grid.dy:
        logger.info(
            "CAUTION:: Structured grids with unequal grid spaces not yet supported"
        )
        assert "Structured grids with unequal grid spaces not yet supported"
    cell_size = grid.values.copy()
    sz = cell_size.shape
    sz = (sz[0], sz[1], 1)
    cell_size = cell_size.flatten("F")
    tmp = gradient_limit([*sz], elen, gradation, 10000, cell_size)
    tmp = np.reshape(tmp, (sz[0], sz[1]), "F")
    grid_limited = Grid(
        bbox=grid.bbox,
        dx=grid.dx,
        values=tmp,
        hmin=grid.hmin,
        extrapolate=grid.extrapolate,
        crs=crs,
    )
    grid_limited.build_interpolant()
    return grid_limited


def distance_sizing_function(
    shoreline,
    rate=0.15,
    max_edge_length=None,
    coarsen=1,
    crs=4326,
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
    coarsen: integer, optional
        Downsample the grid by a constant factor in x and y axes
    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value
    """
    logger.info("Building a distance sizing function...")

    grid = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0 * coarsen,
        hmin=shoreline.h0,
        extrapolate=True,
        values=0.0,
        crs=crs,
    )
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = np.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    points = np.vstack((shoreline.inner, shoreline.mainland))
    # remove shoreline components outside the shoreline.boubox
    boubox = shoreline.boubox
    e_box = edges.get_poly_edges(boubox)
    mask = np.ones((grid.nx, grid.ny), dtype=bool)
    if len(points) > 0:
        try:
            in_boubox, _ = inpoly2(points, boubox, e_box)
            points = points[in_boubox]

            qpts = np.column_stack((lon.flatten(), lat.flatten()))
            in_boubox, _ = inpoly2(qpts, shoreline.boubox, e_box)
            mask_indices = grid.find_indices(qpts[in_boubox, :], lon, lat)
            mask[mask_indices] = False
        except (Exception,):
            ...

    # find location of points on grid
    indices = grid.find_indices(points, lon, lat)
    phi[indices] = -1.0
    try:
        dis = np.abs(skfmm.distance(phi, [grid.dx, grid.dy]))
    except ValueError:
        print("0-level set not found in domain")
        dis = np.zeros((grid.nx, grid.ny)) + 999
    tmp = shoreline.h0 + dis * rate
    if max_edge_length is not None:
        tmp[tmp > max_edge_length] = max_edge_length

    grid.values = np.ma.array(tmp, mask=mask)
    grid.build_interpolant()
    return grid


def feature_sizing_function(
    shoreline,
    signed_distance_function,
    r=3,
    max_edge_length=None,
    plot=False,
    crs=4326,
):
    """Mesh sizes vary proportional to the width or "thickness" of the shoreline
    Implements roughly the approximate medial axis calculation from Eq. 7.1:

    "A MATLAB MESH GENERATOR FOR THE TWO-DIMENSIONAL FINITE ELEMENT METHOD" by Jonas Koko

    Parameters
    ----------
    shoreline: :class:`Shoreline`
        Data processed from :class:`Shoreline`.
    signed_distance_function: a function
        A `signed_distance_function` object
    r: float, optional
        The number of times to divide the shoreline thickness/width to calculate
        the local element size.
    plot: boolean, optional
        Visualize the medial points ontop of the shoreline

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """

    logger.info("Building a feature sizing function...")

    assert r > 0, "local feature size "
    grid_calc = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0 / 2,  # dx is half that of the original shoreline spacing
        hmin=shoreline.h0,
        values=0.0,
        extrapolate=True,
        crs=crs,
    )
    grid = Grid(
        bbox=shoreline.bbox,
        dx=shoreline.h0,
        hmin=shoreline.h0,
        values=0.0,
        extrapolate=True,
        crs=crs,
    )
    # create phi (-1 where shoreline point intersects grid points 1 elsewhere)
    phi = np.ones(shape=(grid_calc.nx, grid_calc.ny))
    lon, lat = grid_calc.create_grid()
    points = np.vstack((shoreline.inner, shoreline.mainland))
    # find location of points on grid
    indices = grid_calc.find_indices(points, lon, lat)
    phi[indices] = -1.0
    dis = np.abs(skfmm.distance(phi, [grid_calc.dx, grid_calc.dy]))
    # calculate the sign
    qpts = np.column_stack((lon.flatten(), lat.flatten()))
    sgn = np.sign(signed_distance_function.eval(qpts))
    sgn = sgn.reshape(*dis.shape)
    dis *= sgn
    gradx, grady = np.gradient(dis, grid_calc.dx, grid_calc.dy)
    grad_mag = np.sqrt(gradx ** 2 + grady ** 2)
    indicies_medial_points = np.where((grad_mag < 0.9) & (dis < -grid_calc.dx))
    medial_points_x, medial_points_y = (
        lon[indicies_medial_points],
        lat[indicies_medial_points],
    )
    medial_points = np.column_stack((medial_points_x, medial_points_y))
    # prune the points twice over to remove spurious medial points
    # (seems to work better than once)
    for _ in range(2):
        medial_points = _prune(medial_points, grid_calc.dx)

    if plot:
        import matplotlib.pyplot as plt

        plt.plot(medial_points[:, 0], medial_points[:, 1], "r.", label="medial points")
        plt.plot(points[:, 0], points[:, 1], "b-", label="shoreline boundaries")
        plt.plot(
            shoreline.boubox[:, 0],
            shoreline.boubox[:, 1],
            "k--",
            label="bounding extents",
        )
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend()
        plt.show()

    # calculate distance to medial axis
    tree = scipy.spatial.cKDTree(medial_points)
    try:
        dMA, _ = tree.query(qpts, k=1, workers=-1)
    except (Exception,):
        dMA, _ = tree.query(qpts, k=1, n_jobs=-1)
    dMA = dMA.reshape(*dis.shape)
    W = dMA + np.abs(dis)
    feature_size = (2 * W) / r

    grid_calc.values = feature_size
    grid_calc.build_interpolant()
    # interpolate the finer grid used for calculations to the final coarser grid
    grid = grid_calc.interpolate_to(grid)
    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.hmin = shoreline.h0
    grid.extrapolate = True
    grid.medial_points = medial_points
    grid.build_interpolant()
    return grid


def _prune(points, dx):
    # Prune medial points
    # Note: We want a line of points larger than around 7.
    # This corresponds to having three points back or forward
    # up the line. Let's check for those three closest points
    # and ensure tehy are within about co*h0 distance from
    # each other where co is the cutoff distance = 0.75*sqrt(2)
    tree = scipy.spatial.cKDTree(points)
    co = 0.75 * np.sqrt(2)
    # build a KDtree w/ the medial points
    try:
        dmed, _ = tree.query(points, k=4, workers=-1)
    except (Exception,):
        dmed, _ = tree.query(points, k=4, n_jobs=-1)
    prune = np.where(
        (dmed[:, 1] > co * dx) | (dmed[:, 2] > 2 * co * dx) | (dmed[:, 3] > 3 * co * dx)
    )
    points = np.delete(points, prune, axis=0)
    return points


def wavelength_sizing_function(
    dem,
    wl=10,
    min_edge_length=None,
    max_edge_length=None,
    verbose=True,
    crs=4326,
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

    Returns
    -------
    :class:`Grid` object
        A sizing function that takes a point and returns a value

    """
    logger.info("Building a wavelength sizing function...")

    lon, lat = dem.create_grid()
    tmpz = dem.eval((lon, lat))

    grav = 9.807
    period = 12.42 * 3600  # M2 period in seconds
    grid = Grid(
        bbox=dem.bbox, dx=dem.dx, dy=dem.dy, extrapolate=True, values=0.0, crs=crs
    )
    tmpz[np.abs(tmpz) < 1] = 1  # Limit minimum depth to 1 m
    grid.values = period * np.sqrt(grav * np.abs(tmpz)) / wl
    grid.values /= 2e6  # to convrt from m to L_R (this needs to be taken out when commiting to github

    if min_edge_length is None:
        grid.hmin = np.amin(grid.values)

    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.build_interpolant()

    return grid


def slope_sizing_function(
    dem,
    slp=20,
    fl=-50,
    min_edge_length=None,
    max_edge_length=None,
    verbose=True,
    crs=4326,
):
    """Mesh sizes that vary proportional to an estimate of the wavelength
       of the M2 tidal constituent

    Parameters
    ----------
    dem:  :class:`Dem`
        Data processed from :class:`Dem`.
    fl: float, optional
        The filter equal to Rossby radius divided by fl
    slp: integer, optional
        The number of modes to resolve slope gradients
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
    from oceanmesh.filterfx import filt2

    if verbose > 0:
        logger.info("Building a slope length sizing function...")
    x, y = dem.create_grid()
    tmpz = dem.eval((x, y))
    grid = Grid(
        bbox=dem.bbox, dx=dem.dx, dy=dem.dy, extrapolate=True, values=0.0, crs=crs
    )
    x0, xN, y0, yN = dem.bbox
    grav, Rre = 9.807, 7.29e-5  # Gravity and Rotation rate of Earth in radians
    # per second
    tmpz[np.abs(tmpz) < 50] = 50

    try:
        len(fl)

    except TypeError:
        fl = np.array([fl])

    try:
        len(slp)

    except TypeError:
        slp = np.array([slp])

    dx, dy = dem.dx, dem.dy  # for gradient function
    nx, ny = dem.nx, dem.ny
    grid = (nx, ny, dx, dy)
    xg, yg = dem.create_grid()
    coords = (xg, yg)
    tmpz_f = np.zeros(tmpz.shape)

    # Now filtering bathymetry to obtain only relevant features
    # Loop through each set of bandpass filter lengths
    if fl[0] < 0 and fl[0] != -999:
        logger.info("Rossby radius of deformation filter is on.")
        rbfilt = abs(fl[0])
        barot = True
        if hasattr(slp, "__len__"):
            if len(slp) > 1 and slp[1] < 0:
                logger.info("Using 1st-mode baroclinic Rossby radius.")
                barot = False

        else:
            logger.info("Using barotropic Rossby radius")

        fl = []
        filtit = True

    elif fl[0] == 0:
        logger.info("Slope filter is off.")
        fl = []
        tmpz_f[:] = tmpz[:]
        filtit = False

    elif fl[0] == -999:
        logger.info("Slope filter is LEGACY.")
        fl = []
        tmpz_f[:] = tmpz[:]
        filtit = -999

    else:
        filtit = False
        for slp in fl.T:
            if isinstance(slp, (int, float)):
                # Do a low-pass filter
                tmpz_ft = filt2(tmpz, dy, slp, "lp")

            elif slp[1] == 0:
                tmpz_ft = filt2(tmpz, dy, slp[0], "lp")

            elif np.all(slp != 0):
                # Do a bandpass filter
                tmpz_ft = filt2(tmpz, dy, slp, "bp")

            else:
                # Highpass filter not recommended
                print(
                    "Warning:: Highpass filter on bathymetry in slope - \
edgelength function in not recommended"
                )
                tmpz_ft = filt2(tmpz, dy, slp[1], "hp")

            tmpz_f += tmpz_ft

    # Performs bandpass filtering
    if filtit:
        bs, time_taken = rossby_filter(tmpz, dem.bbox, grid, coords, rbfilt, barot)


        # legacy filter
    elif filtit == -999:
        from math import sqrt

        bs = np.empty((nx, ny))
        bs[:] = np.nan
        # Rossby radius of deformation filter
        f = 2 * Rre * abs(np.sin(yg * np.pi / 180))  # Local Coriolis coefficient
        # limit to 1000 km
        rosb = np.minimum(
            1000e3, sqrt(grav * abs(tmpz)) / f
        )  # Gives local Rossby radius everywhere
        # autmatically divide into discrete bins
        _, edges = np.histogram(rosb)
        tmpz_ft = tmpz
        dyb = dy
        # get slope from filtered bathy for the segment only
        by, bx = EarthGradient(tmpz_ft, dy, dx)  # get slope in x and y directions
        tempbs = np.sqrt(bx ** 2 + by ** 2)
        # get overall slope
        for i in range(len(edges) - 1):
            sel = (rosb >= edges[i]) & (rosb <= edges[i + 1])
            rosbylb = np.mean(edges[i : i + 1])

            if rosbylb > 2 * dyb:
                tmpz_ft = filt2(tmpz_ft, dyb, rosbylb, "lp")
                dyb = rosbylb

                # get slope from filtered bathy for the segment only
                by, bx = EarthGradient(
                    tmpz_ft, dy, dx
                )  # get slope in x and y directions
                tempbs = np.sqrt(bx ** 2 + by ** 2)  # get overall slope

            else:
                # otherwise just use the same tempbs from before
                pass

            # put in the full one
            bs[sel] = tempbs[sel]

    else:
        # get slope from (possibly filtered) bathy
        by, bx = EarthGradient(tmpz_f, dy, dx)  # get slope in x and y directions
        bs = np.sqrt(bx ** 2 + by ** 2)  # get overall slope

    del bx, by

    # Allow user to specify depth ranges for slope parameter.
    slpd = np.empty((nx, ny))
    slpd[:] = np.nan

    for param in slp.T:
        if not hasattr(param, "__len__"):
            # no bounds specified. valid in this range.
            slpp = param
            z_min = np.NINF
            z_max = np.inf

        else:
            slpp, z_min, zmax = param[:3]

        # Calculating the slope function
        eps = 1e-10  # small number to approximate derivative
        dp = np.maximum(1, tmpz)
        tslpd = (2 * np.pi / slpp) * dp / (bs + eps)
        # apply slope with mask
        limidx = (tmpz >= z_min) & (tmpz < z_max)
        slpd[limidx] = tslpd[limidx]
        del tslpd

    del tmpz, xg, yg
    grid.values = slpd

    if min_edge_length is None:
        grid.hmin = np.amin(grid.values)

    if max_edge_length is not None:
        grid.values[grid.values > max_edge_length] = max_edge_length

    grid.hmin = np.min(grid.values)
    grid.build_interpolant()

    return grid

def rossby_filter(tmpz, bbox, grid, coords, rbfilt, barot):
    """
    Performs the Rossby radius filtering if filtit==True in
    slope_sizing_function.

    Parameters
    ----------
    tmpz : numpy.ndarray
        Contains the bathymetric data across the grid formed by coordinate
        arrays (xg, yg).
    bbox : tuple
        Describes the boundary box of our domain.
    grid : tuple
        Contains the information regarding normals and grid resolutions,
        (nx, ny, dx, dy).
    coords : tuple np.ndarray
        A tuple of two numpy.ndarray describing the longitude and latitude
        coordinate system of our grid.
    rbfilt : float
        Describes the corresponding rossby radius to filter out
    barot : bool
        If True, the function uses the barotropic Rossby radius of deformation.

    Returns
    -------
    bs : numpy.ndarray
        This is essentially grad(h) squared after performing the bandpass
        filtering on the Rossby radius of deformation.
    time_taken : float
        the time taken to prform the filtering process.

    """
    import time
    import math
    from filterfx import filt2

    x0, xN, y0, yN = bbox
    nx, ny, dx, dy = grid
    xg, yg = coords

    start = time.perf_counter()
    bs = np.empty(tmpz.shape)
    bs[:] = np.nan

    # Break into 10 deg latitude chuncsm or less if higher resolution
    div = math.ceil(min(1e7 / nx, 10 * ny / (yN - y0)))
    grav, Rre = 9.807, 7.29e-5  # Gravity and Rotation rate of Earth in radians
    # per second
    nb = math.ceil(ny / div)
    n2s = 0

    for jj in range(nb):
        n2e = min(ny, n2s + div - 1)
        # Rossby radius of deformation filter
        # See Shelton, D. B., et al. (1998): Geographical variability of the
        # first-baroclinic Rossby radius of deformation. J. Phys. Oceanogr.,
        # 28, 433-460.
        ygg = yg[:, n2s:n2e]
        dxx = np.mean(np.diff(xg[n2s:n2e, 0]))
        f = 2 * Rre * abs(np.sin(ygg * np.pi / 180))
        if barot:
            # Barotropic case
            c = np.sqrt(grav * np.maximum(1, -tmpz[:, n2s:n2e]))

        else:
            # Baroclinic case (estimate Nm to be 2.5e-3)
            Nm = 2.5e-3  # Δz x N, where N is Brunt-Vaisala frequency,
            # sqrt(-g/ρ0 * dρ/dz), giving sqrt(-g * (Δρ/ρ0) * Δz)
            c = Nm * np.maximum(1, -tmpz[:, n2s:n2e]) / np.pi

        rosb = c / f
        # Update for equatorial regions
        indices = abs(ygg) < 5
        Re = 6.371e6  # Earth radius at equator in SI units of metres
        twobeta = 4 * Rre * np.cos(ygg[indices] * np.pi / 180) / Re
        rosb[indices] = np.sqrt(c[indices] / twobeta)
        # limit rossby radius to 10,000 km for practical purposes
        rosb[rosb > 1e7] = 1e7
        # Keep lengthscales rbfilt * barotropic
        # radius of deformation
        rosb = np.minimum(10, np.maximum(0, np.floor(np.log2(rosb / dy / rbfilt))))
        edges = np.unique(np.copy(rosb))
        bst = rosb * 0
        for i in range(len(edges)):
            if edges[i] > 0:
                mult = 2 ** edges[i]
                xl, xu = 1, nx
                if ((np.max(xg) > 179 and np.min(xg) < -179)) or (
                    np.max(xg) > 359 and np.min(xg) < 1
                ):
                    # wraps around
                    logger.info("wrapping around")
                    xr = np.concatenate(
                        [
                            np.arange(nx - mult / 2, nx, 1),
                            np.arange(xl, xu),
                            np.arange(1, mult / 2),
                        ],
                        dtype=int,
                    )
                else:
                    xr = np.arange(xl - 1, xu, dtype=int)

                yl, yu = max(1, n2s - mult / 2), min(ny, n2e + mult / 2)
                if np.max(yg) > 89 and yu == ny:
                    # create mirror around pole
                    yr = np.concatenate(
                        [
                            np.arange(yl, yu),
                            np.arange(yu - 1, 2 * ny - n2e - mult / 2, -1),
                        ],
                        dtype=int,
                    )
                else:
                    yr = np.arange(yl - 1, yu, dtype=int)

                xr, yr = xr[:, None], yr[None, :]

                if mult == 2:
                    tmpz_ft = filt2(tmpz[xr, yr], min([dxx, dy]), dy * 2.01, "lp")
                else:
                    tmpz_ft = filt2(tmpz[xr, yr], min([dxx, dy]), dy * mult, "lp")

                # delete the padded region
                tmpz_ft[: np.where(xr == 1)[0][0], :] = 0
                tmpz_ft[nx:, :] = 0
                tmpz_ft[:, : np.where(yr == n2s)[0][0]] = 0
                tmpz_ft[:, n2e - n2s + 2 :] = 0

            else:
                tmpz_ft = tmpz[:, n2s:n2e]

            by, bx = EarthGradient(
                tmpz_ft, dy, dx
            )  # [n2s:n2e]) # get slope in x and y directions
            tempbs = np.sqrt(bx ** 2 + by ** 2)  # get overall slope
            bst[rosb == edges[i]] = tempbs[rosb == edges[i]]

        bs[:, n2s:n2e] = bst
        n2s = n2e + 1

    time_taken = time.perf_counter() - start

    return bs, time_taken

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
        logger.info(f"For sizing function #{idx1}")

        # interpolate all finer nests onto coarse func and enforce gradation rate
        for k, finer in enumerate(list_of_grids[idx1 + 1 :]):
            logger.info(
                f"  Interpolating sizing function #{idx1+1 + k} onto sizing function #{idx1}"
            )
            _wkt = finer.crs.to_dict()
            if "units" in _wkt:
                _dx = finer.dx
            else:
                # it must be degrees?
                _dx = finer.dx * 111e3
            _blend_width = int(np.floor(blend_width / _dx))
            finer.extrapolate = False
            new_coarse = finer.blend_into(
                new_coarse, blend_width=_blend_width, p=p, nnear=nnear
            )
            new_coarse.extrapolate = True
            new_coarse.build_interpolant()
        # append it to list
        new_list_of_grids.append(new_coarse)

    list_of_grids[-1].extrapolate = True

    # retain the finest
    new_list_of_grids.append(list_of_grids[-1])

    # return the mesh size function to query during genertaion
    # NB: only keep the minimum value over all grids
    def func(qpts):
        hmin = np.array([999999] * len(qpts))
        for i, grid in enumerate(new_list_of_grids):
            if i == 0:
                grid.extrapolate = True
            else:
                grid.extrapolate = False
            grid.build_interpolant()
            _hmin = grid.eval(qpts)
            hmin = np.min(np.column_stack([_hmin, hmin]), axis=1)
        return hmin

    return func, new_list_of_grids


def EarthGradient(F, dy, dx):
    """
    EarthGradient(F,HX,HY), where F is 2-D, uses the spacing
    specified by HX and HY. HX and HY can either be scalars to specify
    the spacing between coordinates or vectors to specify the
    coordinates of the points.  If HX and HY are vectors, their length
    must match the corresponding dimension of F.
    """
    Fy, Fx = np.zeros(F.shape), np.zeros(F.shape)

    # Forward diferences on edges
    Fx[:, 0] = (F[:, 1] - F[:, 0]) / dx
    Fx[:, -1] = (F[:, -1] - F[:, -2]) / dx
    Fy[0, :] = (F[1, :] - F[0, :]) / dy
    Fy[-1, :] = (F[-1, :] - F[-2, :]) / dy

    # Central Differences on interior
    Fx[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (2 * dx)
    Fy[1:-1, :] = (F[2:, :] - F[:-2, :]) / (2 * dy)

    return Fy, Fx
