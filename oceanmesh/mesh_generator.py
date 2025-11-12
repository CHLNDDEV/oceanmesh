import datetime
import logging
import os
import time

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import scipy.sparse as spsparse
from _delaunay_class import DelaunayTriangulation as DT
from _fast_geometry import unique_edges
from pyproj import CRS

from .clean import _external_topology
from .edgefx import multiscale_sizing_function
from .fix_mesh import fix_mesh
from .grid import Grid
from .region import (
    to_lat_lon,
    to_stereo,
    bbox_contains,
    validate_crs_compatible,
    get_crs_string,
    is_global_bbox,
)
from .signed_distance_function import Domain, multiscale_signed_distance_function

logger = logging.getLogger(__name__)

__all__ = [
    "generate_mesh",
    "generate_multiscale_mesh",
    "plot_mesh_connectivity",
    "plot_mesh_bathy",
    "write_to_fort14",
    "write_to_t3s",
]


def write_to_fort14(
    points,
    cells,
    filepath,
    topobathymetry=None,
    project_name="Created with oceanmesh",
    flip_bathymetry=False,
):
    """
    Parameters
    -----------
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the fort.14 file to.
    topobathymetry (numpy.ndarray): An array of shape (np, 1) containing the topobathymetry values at each node.
    project_name (str): The name of the project to be written to the fort.14 file.
    flip_bathymetry (bool): If True, the bathymetry values will be multiplied by -1.

    Returns:
    --------
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the fort.14 file to.
    """
    logger.info("Exporting mesh to fort.14 file...")

    # Calculate number of nodes and elements
    npoints = np.size(points, 0)
    nelements = np.size(cells, 0)

    if topobathymetry is not None:
        assert (
            len(topobathymetry) == npoints
        ), "topobathymetry must be the same length as points"
    else:
        topobathymetry = np.zeros((npoints, 1))

    if flip_bathymetry:
        topobathymetry *= -1

    # Shift cell indices by 1 (fort.14 uses 1-based indexing)
    cells += 1

    # Open file for writing
    with open(filepath, "w") as f_id:
        # Write mesh name
        if flip_bathymetry:
            f_id.write(f"{project_name} (bathymetry flipped) \n")
        else:
            f_id.write(f"{project_name} \n")

        # Write number of nodes and elements
        np.savetxt(
            f_id,
            np.column_stack((nelements, npoints)),
            delimiter=" ",
            fmt="%i",
            newline="\n",
        )

        # Write node coordinates
        for k in range(npoints):
            np.savetxt(
                f_id,
                np.column_stack((k + 1, points[k][0], points[k][1], topobathymetry[k])),
                delimiter=" ",
                fmt="%i %f %f %f",
                newline="\n",
            )

        # Write element connectivity
        for k in range(nelements):
            np.savetxt(
                f_id,
                np.column_stack((k + 1, 3, cells[k][0], cells[k][1], cells[k][2])),
                delimiter=" ",
                fmt="%i %i %i %i %i ",
                newline="\n",
            )

        # Write zero for each boundary condition (4 total)
        for k in range(4):
            f_id.write("%d \n" % 0)

    return f"Wrote the mesh to {filepath}..."


def write_to_t3s(points, cells, filepath):
    """
    Write mesh data to a t3s file.

    Parameters:
    points (numpy.ndarray): An array of shape (np, 2) containing the x, y coordinates of the mesh nodes.
    cells (numpy.ndarray): An array of shape (ne, 3) containing the indices of the nodes that form each mesh element.
    filepath (str): The file path to write the t3s file to.

    Returns:
    None
    """
    logger.info("Exporting mesh to t3s file...")

    # Calculate number of nodes and elements
    npoints = np.size(points, 0)
    nelements = np.size(cells, 0)

    # Open file for writing
    with open(filepath, "w") as f_id:
        # Write header
        today = datetime.datetime.now()
        date_time = today.strftime("%m/%d/%Y, %H:%M:%S")
        t3head = (
            """#########################################################################\n
        :FileType t3s ASCII EnSim 1.0\n
        # Canadian Hydraulics Centre/National Research Council (c) 1998-2004\n
        # DataType 2D T3 Scalar Mesh\n
        #
        :Application BlueKenue\n
        :Version 3.0.44\n
        :WrittenBy pyoceanmesh\n
        :CreationDate """
            + date_time
            + """\n
        #
        #------------------------------------------------------------------------\n
        #
        :Projection Cartesian\n
        :Ellipsoid Unknown\n
        #
        :NodeCount """
            + str(npoints)
            + """\n
        :ElementCount """
            + str(nelements)
            + """\n
        :ElementType T3\n
        #
        :EndHeader"""
        )  # END HEADER
        t3head = os.linesep.join([s for s in t3head.splitlines() if s])
        f_id.write(t3head)
        f_id.write("\n")

        # Write node coordinates
        for k in range(npoints):
            np.savetxt(
                f_id,
                np.column_stack((points[k][0], points[k][1], 0.0)),
                delimiter=" ",
                fmt="%f %f %f",
                newline="\n",
            )

        # Write element connectivity
        for k in range(nelements):
            np.savetxt(
                f_id,
                np.column_stack((cells[k][0], cells[k][1], cells[k][2])),
                delimiter=" ",
                fmt="%i %i %i ",
                newline="\n",
            )

    return f"Wrote the mesh to {filepath}..."


def plot_mesh_connectivity(points, cells, show_plot=True):
    """Plot the mesh connectivity using matplotlib's triplot function.
    Parameters
    ----------

    points : numpy.ndarray
        A 2D array containing the x and y coordinates of the points.
    cells : numpy.ndarray
        A 2D array containing the connectivity information for the triangles.
    show_plot : bool, optional
        Whether to show the plot or not. The default is True.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.triplot(triang, lw=0.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Mesh connectivity")
    if show_plot:
        plt.show(block=False)
    return ax


def plot_mesh_bathy(points, bathymetry, connectivity, show_plot=True):
    """
    Create a tricontourf plot of the bathymetry data associated with the points,
    using the triangle connectivity information to plot the contours.

    Parameters
    ----------
    points : numpy.ndarray
        A 2D array containing the x and y coordinates of the points.
    bathymetry : numpy.ndarray
        A 1D array containing the bathymetry values associated with each point.
    connectivity : numpy.ndarray
        A 2D array containing the connectivity information for the triangles.
    show_plot : bool, optional
        Whether or not to display the plot. Default is True.

    Returns
    -------
    matplotlib.axes._subplots.AxesSubplot
        The axis handle of the plot.

    """
    # Create a Triangulation object using the points and connectivity table
    triangulation = tri.Triangulation(points[:, 0], points[:, 1], connectivity)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the tricontourf
    tricontourf = ax.tricontourf(triangulation, bathymetry, cmap="jet")

    # Add colorbar
    plt.colorbar(tricontourf)

    # Set axis labels
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Set title
    ax.set_title("Mesh Topobathymetry")

    # Show the plot if requested
    if show_plot:
        plt.show()

    return ax


def _parse_kwargs(kwargs):
    for key in kwargs:
        if key in {
            "nscreen",
            "max_iter",
            "seed",
            "pfix",
            "points",
            "domain",
            "edge_length",
            "bbox",
            "min_edge_length",
            "plot",
            "blend_width",
            "blend_polynomial",
            "blend_max_iter",
            "blend_nnear",
            "lock_boundary",
            "pseudo_dt",
            "stereo",
        }:
            pass
        else:
            raise ValueError(
                "Option %s with parameter %s not recognized " % (key, kwargs[key])
            )


def _check_bbox(bbox):
    assert isinstance(bbox, tuple), "`bbox` must be a tuple"
    assert int(len(bbox) / 2), "`dim` must be 2"


def _validate_multiscale_domains(domains, edge_lengths):
    """Validate domain & sizing function compatibility for multiscale meshing.

    Checks performed:
      1. Presence of CRS on all domains.
      2. Global domain (stereo=True) ordering: must be first if present.
      3. Bbox containment for regional domains within global domain when mixing.
      4. CRS compatibility (global EPSG:4326, regional geographic or projected).
      5. Edge length Grid CRS matches corresponding domain CRS.
      6. Stereo flag usage: only global domain should have stereo=True.

    Returns
    -------
    (ok: bool, errors: list[str])
    """
    errors = []

    if len(domains) != len(edge_lengths):
        errors.append("Number of domains and edge_lengths differ.")
        return False, errors

    # Determine global domain (stereo=True)
    stereo_flags = [getattr(d, "stereo", False) for d in domains]
    global_indices = [i for i, s in enumerate(stereo_flags) if s]
    global_domain = None
    if len(global_indices) > 1:
        errors.append("Only one global (stereo=True) domain permitted.")
    elif len(global_indices) == 1:
        if global_indices[0] != 0:
            errors.append("Global domain must be the first (coarsest) domain in list.")
        global_domain = domains[global_indices[0]]

    # Helper flags for conditional CRS requirements
    domain_crs_list = [getattr(d, "crs", None) for d in domains]
    edge_crs_list = [getattr(el, "crs", None) if hasattr(el, "crs") else None for el in edge_lengths]
    has_any_crs = any(c is not None for c in domain_crs_list) or any(c is not None for c in edge_crs_list)

    # Detect implicit global domain: EPSG:4326 + global-like bbox but stereo=False
    implicit_global_idx = None
    for i, d in enumerate(domains):
        dcrs = getattr(d, "crs", None)
        try:
            if dcrs is not None and CRS.from_user_input(dcrs).to_epsg() == 4326 and is_global_bbox(d.bbox):
                implicit_global_idx = i
                break
        except Exception:
            # If CRS can't be parsed, skip implicit detection for this domain
            pass

    # If no CRS anywhere and no implicit global detected, allow bbox-only flows without error
    if has_any_crs or implicit_global_idx is not None:
        # Now missing CRS becomes an error because compatibility checks are required
        for i, d in enumerate(domains):
            if getattr(d, "crs", None) is None:
                errors.append(
                    f"Domain #{i} missing CRS metadata. Provide CRS via Shoreline(crs=...) to enable compatibility checks."
                )

    # If we have a global domain, validate CRS and containment
    if global_domain is not None:
        gcrs = getattr(global_domain, "crs", None)
        if gcrs is not None:
            try:
                gcrs_str = get_crs_string(gcrs)
            except Exception:
                gcrs_str = str(gcrs)
            # Explicitly require global CRS be EPSG:4326
            try:
                parsed = CRS.from_user_input(gcrs)
                if parsed.to_epsg() != 4326:
                    errors.append(
                        f"Global domain CRS {gcrs_str} must be EPSG:4326 for global+regional multiscale meshing."
                    )
            except Exception:
                errors.append(
                    f"Global domain CRS '{gcrs_str}' could not be parsed; expected EPSG:4326."
                )
        # Containment checks for regional domains
        for i, d in enumerate(domains[1:], start=1):
            # Perform containment in lat/lon if global stereographic domain provided its bbox in stereo space
            g_bbox = global_domain.bbox
            d_bbox = d.bbox
            try:
                if getattr(global_domain, 'stereo', False):
                    # Convert regional bbox corners to stereo then compare
                    lon_min, lon_max, lat_min, lat_max = d_bbox
                    reg_corners_lon = [lon_min, lon_max, lon_max, lon_min]
                    reg_corners_lat = [lat_min, lat_min, lat_max, lat_max]
                    sx, sy = to_stereo(np.array(reg_corners_lon), np.array(reg_corners_lat))
                    stereo_reg_bbox = (float(np.min(sx)), float(np.max(sx)), float(np.min(sy)), float(np.max(sy)))
                    if not bbox_contains(g_bbox, stereo_reg_bbox):
                        errors.append(
                            f"Regional domain #{i} bbox {d_bbox} (lat/lon) not contained within global stereo bbox {g_bbox}."
                        )
                else:
                    if not bbox_contains(g_bbox, d_bbox):
                        errors.append(
                            f"Regional domain #{i} bbox {d_bbox} not contained within global bbox {g_bbox}."
                        )
            except Exception:
                errors.append(
                    f"Regional domain #{i} containment check failed due to transformation error; verify CRS and stereo settings."
                )
            # Stereo flag must be False for regional domains
            if getattr(d, "stereo", False):
                errors.append(f"Regional domain #{i} has stereo=True; only the global domain may set stereo=True.")
            # CRS compatibility between global and regional
            ok_crs, msg_crs = validate_crs_compatible(getattr(global_domain, "crs", None), getattr(d, "crs", None))
            if not ok_crs:
                errors.append(msg_crs)

    # Implicit global-like domain with EPSG:4326 but stereo=False mixing with different CRS
    if implicit_global_idx is not None:
        ig = domains[implicit_global_idx]
        if not getattr(ig, "stereo", False):
            # If any other domain has a CRS that is not equal to EPSG:4326, require stereo=True and ordering
            try:
                ig_crs = CRS.from_user_input(ig.crs) if getattr(ig, "crs", None) is not None else None
                for j, d in enumerate(domains):
                    if j == implicit_global_idx:
                        continue
                    dcrs = getattr(d, "crs", None)
                    if dcrs is None:
                        continue
                    if ig_crs is None:
                        continue
                    if not ig_crs.equals(CRS.from_user_input(dcrs)):
                        errors.append(
                            "Detected global-like EPSG:4326 domain without stereo=True mixed with other CRS. "
                            "Set stereo=True on the global domain and place it first in the list."
                        )
                        break
            except Exception:
                # If CRS parsing fails, skip this implicit enforcement
                pass

    # Edge length CRS matching
    for i, (d, el) in enumerate(zip(domains, edge_lengths)):
        if hasattr(el, "crs"):
            el_crs = getattr(el, "crs", None)
            d_crs = getattr(d, "crs", None)
            if el_crs is not None and d_crs is not None:
                try:
                    if not CRS.from_user_input(el_crs).equals(CRS.from_user_input(d_crs)):
                        errors.append(
                            f"Edge length #{i} CRS {get_crs_string(el_crs)} does not match domain CRS {get_crs_string(d_crs)}."
                        )
                except Exception:
                    errors.append(
                        f"Edge length #{i} CRS could not be compared to domain CRS (el={get_crs_string(el_crs)}, domain={get_crs_string(d_crs)})."
                    )

    return len(errors) == 0, errors


# NOTE: stereo-aware sizing wrapper removed per verification comment; sizing
# functions are always evaluated on lat/lon points supplied by generate_mesh.


def generate_multiscale_mesh(domains, edge_lengths, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to several
    sizing functions `edge_lengths` and several signed distance functions
    See the kwargs for `generate_mesh`.

    This function supports both regional multiscale meshing (multiple nested
    domains in the same projection) and global+regional multiscale meshing
    (a global domain in stereographic projection with one or more regional
    refinement zones defined in WGS84). For global+regional workflows,
    coordinate transformations between WGS84 (EPSG:4326) and stereographic
    space are handled automatically during mesh generation. Users define all
    sizing functions on latitude/longitude grids; the mesher manages the
    projection conversions transparently when the first (global) domain has
    `stereo=True`.

    Parameters
    ----------
    domains: A list of function objects.
        A list of functions that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_lengths: A function object.
        A list of functions that can evalulate a point and return a mesh size.
    \**kwargs:
        See below for kwargs in addition to the ones available for `generate_mesh`

    Requirements for mixing global and regional domains
    ---------------------------------------------------
    - Global domain must be first in the list and must use EPSG:4326 with stereo=True
    - Regional domains must not set stereo=True
    - Each regional domain bbox must be fully contained by the global domain bbox
    - All domains and sizing Grid objects must supply CRS metadata; each Grid CRS must match its domain CRS
    - Global+regional CRS mixing supported only when global=EPSG:4326 and regional is geographic or projected
        - Coordinate transformation workflow: sizing functions are defined in EPSG:4326; the global mesh is generated in stereographic space; automatic conversions applied during sizing evaluation
        - The global domain requires two shoreline datasets: one in lat/lon (for sizing functions), one in stereographic (for the meshing boundary)

        Automatic coordinate handling for global+regional meshing
        -----------------------------------------------------------
        When the first domain has `stereo=True`, this function automatically:
            * Detects the global+regional mixing scenario during validation.
            * Transforms query points between stereographic and lat/lon when evaluating regional sizing grids.
            * Applies stereographic distortion corrections to sizing values where needed.
            * Propagates the `stereo=True` flag to the final blending/union mesh generation step.

        This ensures that regional sizing functions (defined in WGS84) interact correctly with a global mesh generated in stereographic space. Users do not need to manually handle coordinate conversions.

        Example
        -------
        See the README section 'Global mesh generation with regional refinement' for
        a complete example demonstrating how to merge a regional mesh (e.g., Australia)
        into a global mesh.

    :Keyword Arguments:
        * *blend_width* (``float``) --
                The width of the element size transition region between nest and parent
        * *blend_polynomial* (``int``) --
                The rate of transition scales with 1/dist^blend_polynomial
        * *blend_max_iter* (``int``) --
                The number of mesh generation iterations to blend the nest and parent.
        * *blend_nnear* (``int``) --
                The number of nearest neighbors in the IDW interpolation.
    * *stereo* (``bool``) --
        Note: The stereo parameter for the final union/blending step is inferred from the domain
        metadata (global domain first with stereo=True). Users typically should not set this
        explicitly for multiscale workflows.

    Notes
    -----
    * Regional-only multiscale meshing (no global domain) requires all domains share a compatible CRS.
    * Global+regional meshing follows a two-step workflow: sizing in WGS84, global meshing in stereographic space.
    * Validation errors provide detailed guidance (CRS mismatches, bbox containment, stereo flag misuse).
    * Domain metadata (CRS, stereo flags) is collected internally to manage automatic coordinate transformations.

    """
    assert (
        len(domains) > 1 and len(edge_lengths) > 1
    ), "This function takes a list of domains and sizing functions"
    assert len(domains) == len(
        edge_lengths
    ), "The same number of domains must be passed as sizing functions"

    # Perform validation prior to any mesh generation steps
    ok, verrors = _validate_multiscale_domains(domains, edge_lengths)
    if not ok:
        formatted = "\n - " + "\n - ".join(verrors)
        raise ValueError(
            "Multiscale domain validation failed with the following issues:" + formatted + "\nGuidance: Ensure a single global domain (stereo=True, EPSG:4326) precedes regional domains; supply CRS metadata via Shoreline; regional bboxes must lie within global bbox; sizing Grid CRS must match domain CRS."
        )
    opts = {
        "max_iter": 100,
        "seed": 0,
        "pfix": None,
        "points": None,
        "min_edge_length": None,
        "plot": 999999,
        "blend_width": 2500,
        "blend_polynomial": 2,
        "blend_max_iter": 20,
        "blend_nnear": 256,
        "lock_boundary": False,
    }
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    # Build domain metadata for stereo/CRS awareness during blending
    domain_metadata = {
        "stereo_flags": [getattr(d, "stereo", False) for d in domains],
        "crs_list": [getattr(d, "crs", None) for d in domains],
        # Consider the first domain as potential global parent
        "global_stereo": bool(getattr(domains[0], "stereo", False)),
    }

    master_edge_length, edge_lengths_smoothed = multiscale_sizing_function(
        edge_lengths,
        blend_width=opts["blend_width"],
        nnear=opts["blend_nnear"],
        p=opts["blend_polynomial"],
        domain_metadata=domain_metadata,
    )
    # Sanitize hmin on each smoothed sizing grid to ensure positivity
    for i, el in enumerate(edge_lengths_smoothed):
        if isinstance(el, Grid):
            hmin = getattr(el, "hmin", None)
            if hmin is None or not np.isfinite(hmin) or hmin <= 0:
                vals = el.values
                if np.ma.isMaskedArray(vals):
                    vals = np.ma.filled(vals, np.nan)
                vals = np.asarray(vals)
                pos = vals[np.isfinite(vals) & (vals > 0)]
                if pos.size > 0:
                    el.hmin = float(np.nanmin(pos))
                    logger.warning(
                        f"Sizing grid #{i} had invalid hmin; recomputed fallback hmin={el.hmin:.3f}"
                    )
                else:
                    raise ValueError(
                        f"Sizing grid #{i} contains no positive values to determine a minimum edge length."
                    )
    union, nests = multiscale_signed_distance_function(domains)
    _p = []
    global_minimum = 9999
    for domain_number, (sdf, edge_length) in enumerate(
        zip(nests, edge_lengths_smoothed)
    ):
        logger.info(f"--> Building domain #{domain_number}")
        global_minimum = np.amin([global_minimum, edge_length.hmin])
        # Use the domain's own stereo flag (global first domain may be stereo=True)
        _tmpp, _ = generate_mesh(
            sdf,
            edge_length,
            stereo=getattr(domains[domain_number], "stereo", False),
            **kwargs,
        )
        _p.append(_tmpp)

    _p = np.concatenate(_p, axis=0)

    # merge the two domains together
    logger.info("--> Blending the domains together...")
    # Avoid passing duplicate max_iter to generate_mesh
    _kwargs = dict(kwargs)
    _kwargs.pop("max_iter", None)
    # If union is global stereo, ensure stereo flag passed to final blending mesh generation
    if getattr(union, "stereo", False):
        _kwargs["stereo"] = True
    _p, _t = generate_mesh(
        domain=union,
        edge_length=master_edge_length,
        min_edge_length=global_minimum,
        points=_p,
        max_iter=opts["blend_max_iter"],
        lock_boundary=True,
        **_kwargs,
    )

    return _p, _t


def generate_mesh(domain, edge_length, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to a
        sizing function `edge_length` and signed distance function.

    Parameters
    ----------
    domain: A function object.
        A function that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_length: A function object.
        A function that can evalulate a point and return a mesh size.
    \**kwargs:
        See below

    :Keyword Arguments:
        * *bbox* (``tuple``) --
            Bounding box containing domain extents. REQUIRED IF NOT USING :class:`edge_length`
        * *max_iter* (``float``) --
            Maximum number of meshing iterations. (default==50)
        * *seed* (``float`` or ``int``) --
            Pseudo-random seed to initialize meshing points. (default==0)
        * *pfix* (`array-like`) --
            An array of points to constrain in the mesh. (default==None)
        * *min_edge_length* (``float``) --
            The minimum element size in the domain. REQUIRED IF NOT USING :class:`edge_length`
        * *plot* (``int``) --
            The mesh is visualized every `plot` meshing iterations.
        * *pseudo_dt* (``float``) --
            The pseudo time step for the meshing algorithm. (default==0.2)
        * *stereo* (``bool``) --
            To mesh the whole world (default==False)

    Returns
    -------
    points: array-like
        vertex coordinates of mesh
    t: array-like
        mesh connectivity table.

    """
    _DIM = 2
    opts = {
        "max_iter": 50,
        "seed": 0,
        "pfix": None,
        "points": None,
        "min_edge_length": None,
        "plot": 999999,
        "lock_boundary": False,
        "pseudo_dt": 0.2,
        "stereo": False,
    }
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    fd, bbox = _unpack_domain(domain, opts)
    fh, min_edge_length = _unpack_sizing(edge_length, opts)

    _check_bbox(bbox)
    bbox = np.array(bbox).reshape(-1, 2)

    assert min_edge_length > 0, "`min_edge_length` must be > 0"

    assert opts["max_iter"] > 0, "`max_iter` must be > 0"
    max_iter = opts["max_iter"]

    np.random.seed(opts["seed"])

    L0mult = 1 + 0.4 / 2 ** (_DIM - 1)
    delta_t = opts["pseudo_dt"]
    geps = 1e-3 * np.amin(min_edge_length)
    deps = np.sqrt(np.finfo(np.double).eps)  # * np.amin(min_edge_length)

    pfix, nfix = _unpack_pfix(_DIM, opts)
    lock_boundary = opts["lock_boundary"]

    if opts["points"] is None:
        p = _generate_initial_points(
            min_edge_length,
            geps,
            bbox,
            fh,
            fd,
            pfix,
            opts["stereo"],
        )
    else:
        p = opts["points"]

    N = p.shape[0]

    assert N > 0, "No vertices to mesh with!"

    logger.info(
        f"Commencing mesh generation with {N} vertices will perform {max_iter} iterations."
    )
    for count in range(max_iter):
        start = time.time()

        # (Re)-triangulation by the Delaunay algorithm
        dt = DT()
        dt.insert(p.ravel().tolist())

        # Get the current topology of the triangulation
        p, t = _get_topology(dt)

        ifix = []
        if lock_boundary:
            _, bpts = _external_topology(p, t)
            for fix in bpts:
                ifix.append(_closest_node(fix, p))
                nfix = len(ifix)

        # Find where pfix went
        if nfix > 0:
            for fix in pfix:
                ind = _closest_node(fix, p)
                ifix.append(ind)
                p[ind] = fix

        # Remove points outside the domain
        t = _remove_triangles_outside(p, t, fd, geps)

        # Number of iterations reached, stop.
        if count == (max_iter - 1):
            p, t, _ = fix_mesh(p, t, dim=_DIM, delete_unused=True)
            logger.info("Termination reached...maximum number of iterations.")
            return p, t

        # Compute the forces on the bars
        Ftot = _compute_forces(p, t, fh, min_edge_length, L0mult, opts)

        # Force = 0 at fixed points
        Ftot[:nfix] = 0

        # Update positions
        p += delta_t * Ftot

        # Bring outside points back to the boundary
        p = _project_points_back(p, fd, deps)

        # Show the user some progress so they know something is happening
        maxdp = delta_t * np.sqrt((Ftot**2).sum(1)).max()

        logger.info(
            f"Iteration #{count + 1}, max movement is {maxdp}, there are {len(p)} vertices and {len(t)}"
        )

        end = time.time()
        logger.info(f"Elapsed wall-clock time {end - start} seconds")


def _unpack_sizing(edge_length, opts):
    if isinstance(edge_length, Grid):
        fh = edge_length.eval
        min_edge_length = edge_length.hmin
        # Defensive: if hmin is invalid, recompute from grid values
        if min_edge_length is None or not np.isfinite(min_edge_length) or min_edge_length <= 0:
            vals = edge_length.values
            if np.ma.isMaskedArray(vals):
                vals = np.ma.filled(vals, np.nan)
            vals = np.asarray(vals)
            pos = vals[np.isfinite(vals) & (vals > 0)]
            if pos.size > 0:
                min_edge_length = float(np.nanmin(pos))
                edge_length.hmin = min_edge_length
                logger.warning(
                    f"Edge length grid had invalid hmin; recomputed fallback min_edge_length={min_edge_length:.3f}"
                )
            else:
                raise ValueError(
                    "Edge length grid contains no positive values to determine a minimum edge length."
                )
    elif callable(edge_length):
        fh = edge_length
        min_edge_length = opts["min_edge_length"]
    else:
        raise ValueError(
            "`edge_length` must either be a function or a `edge_length` object"
        )
    return fh, min_edge_length


def _unpack_domain(domain, opts):
    if isinstance(domain, Domain):
        bbox = domain.bbox
        fd = domain.eval
    elif callable(domain):
        bbox = opts["bbox"]
        fd = domain
    else:
        raise ValueError(
            "`domain` must be a function or a :class:`signed_distance_function object"
        )
    return fd, bbox


def _get_bars(t):
    """Describe each bar by a unique pair of nodes"""
    bars = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return unique_edges(bars)


# Persson-Strang
def _compute_forces(p, t, fh, min_edge_length, L0mult, opts):
    """Compute the forces on each edge based on the sizing function"""
    N = p.shape[0]
    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec**2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    if opts["stereo"]:
        # For global+regional multiscale meshes, this branch handles the global stereo case.
        # Regional sizing functions have been wrapped or transformed earlier so fh(p2)
        # evaluates correctly on lat/lon even though points are maintained in stereo space.
        p1 = p[bars].sum(1) / 2
        x, y = to_lat_lon(p1[:, 0], p1[:, 1])
        p2 = np.asarray([x, y]).T
        hbars = fh(p2) * _stereo_distortion_dist(y)
    else:
        hbars = fh(p[bars].sum(1) / 2)
    # Guard against non-finite or non-positive sizing values that can poison forces
    hbars = np.asarray(hbars, dtype=float)
    valid = np.isfinite(hbars) & (hbars > 0)
    if not np.any(valid):
        raise ValueError("Sizing function returned no positive finite values inside domain.")
    if not np.all(valid):
        repl = np.nanmedian(hbars[valid])
        hbars = np.where(valid, hbars, repl)
    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
    F = L0 - L
    F[F < 0] = 0  # Bar forces (scalars)
    Fvec = (
        F[:, None] / L[:, None].dot(np.ones((1, 2))) * barvec
    )  # Bar forces (x,y components)
    Ftot = _dense(
        bars[:, [0] * 2 + [1] * 2],
        np.repeat([list(range(2)) * 2], len(F), axis=0),
        np.hstack((Fvec, -Fvec)),
        shape=(N, 2),
    )
    return Ftot


# Bossen-Heckbert
# def _compute_forces(p, t, fh, min_edge_length, L0mult):
#    """Compute the forces on each edge based on the sizing function"""
#    N = p.shape[0]
#    bars = _get_bars(t)
#    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
#    L = np.sqrt((barvec ** 2).sum(1))  # L = Bar lengths
#    L[L == 0] = np.finfo(float).eps
#    hbars = fh(p[bars].sum(1) / 2)
#    L0 = hbars * L0mult * (np.nanmedian(L) / np.nanmedian(hbars))
#    LN = L / L0
#    F = (1 - LN ** 4) * np.exp(-(LN ** 4)) / LN
#    Fvec = (
#        F[:, None] / LN[:, None].dot(np.ones((1, 2))) * barvec
#    )  # Bar forces (x,y components)
#    Ftot = _dense(
#        bars[:, [0] * 2 + [1] * 2],
#        np.repeat([list(range(2)) * 2], len(F), axis=0),
#        np.hstack((Fvec, -Fvec)),
#        shape=(N, 2),
#    )
#    return Ftot


def _dense(Ix, J, S, shape=None, dtype=None):
    """
    Similar to MATLAB's SPARSE(I, J, S, ...), but instead returning a
    dense array.
    """

    # Advanced usage: allow J and S to be scalars.
    if np.isscalar(J):
        x = J
        J = np.empty(Ix.shape, dtype=int)
        J.fill(x)
    if np.isscalar(S):
        x = S
        S = np.empty(Ix.shape)
        S.fill(x)

    # Turn these into 1-d arrays for processing.
    S = S.flat
    II = Ix.flat
    J = J.flat
    return spsparse.coo_matrix((S, (II, J)), shape, dtype).toarray()


def _remove_triangles_outside(p, t, fd, geps):
    """Remove vertices outside the domain"""
    pmid = p[t].sum(1) / 3  # Compute centroids
    return t[fd(pmid) < -geps]  # Keep interior triangles


def _project_points_back(p, fd, deps):
    """Project points outsidt the domain back within"""
    d = fd(p)
    ix = d > 0  # Find points outside (d>0)
    if ix.any():

        def _deps_vec(i):
            a = [0] * 2
            a[i] = deps
            return a

        try:
            dgrads = [
                (fd(p[ix] + _deps_vec(i)) - d[ix]) / deps for i in range(2)
            ]  # old method
        except ValueError:  # an error is thrown if all points in fd are outside
            # bbox domain, so instead calulate all fd and then
            # take the solely ones outside domain
            dgrads = [(fd(p + _deps_vec(i)) - d) / deps for i in range(2)]
            dgrads = list(np.array(dgrads)[:, ix])
        dgrad2 = sum(dgrad**2 for dgrad in dgrads)
        dgrad2 = np.where(dgrad2 < deps, deps, dgrad2)
        p[ix] -= (d[ix] * np.vstack(dgrads) / dgrad2).T  # Project
    return p


def _stereo_distortion(lat):
    # we use here Stereographic projection of the sphere
    # from the north pole onto the plane
    # https://en.wikipedia.org/wiki/Stereographic_projection
    lat0 = 90
    ll = lat + lat0
    lrad = ll / 180 * np.pi
    res = 2 / (1 + np.sin(lrad))
    return res


def _stereo_distortion_dist(lat):
    lrad = np.radians(lat)
    # Calculate the scale factor for the stereographic projection
    res = 2 / (1 + np.sin(lrad)) / 180 * np.pi
    return res


def _generate_initial_points(min_edge_length, geps, bbox, fh, fd, pfix, stereo=False):
    """Create initial distribution in bounding box (equilateral triangles)"""
    if stereo:
        bbox = np.array([[-180, 180], [-89, 89]])
    p = np.mgrid[
        tuple(slice(min, max + min_edge_length, min_edge_length) for min, max in bbox)
    ].astype(float)
    if stereo:
        # For global meshes (including mixed global+regional) we generate points in lat/lon,
        # then project to stereo. The sizing function fh has already been wrapped (if needed)
        # to internally transform coordinates back to lat/lon for regional grids.
        # for global meshes in stereographic projections,
        # we need to reproject the points from lon/lat to stereo projection
        # then, we need to rectify their coordinates to lat/lon for the sizing function
        p0 = p.reshape(2, -1).T
        x, y = to_stereo(p0[:, 0], p0[:, 1])
        p = np.asarray([x, y]).T
        r0 = fh(to_lat_lon(p[:, 0], p[:, 1])) * _stereo_distortion(p0[:, 1])
    else:
        p = p.reshape(2, -1).T
        r0 = fh(p)
    r0m = np.min(r0[r0 >= min_edge_length])
    p = p[np.random.rand(p.shape[0]) < r0m**2 / r0**2]
    p = p[fd(p) < geps]  # Keep only d<0 points
    return np.vstack(
        (
            pfix,
            p,
        )
    )


def _dist(p1, p2):
    """Euclidean distance between two sets of points"""
    return np.sqrt(((p1 - p2) ** 2).sum(1))


def _unpack_pfix(dim, opts):
    """Unpack fixed points"""
    pfix = np.empty((0, dim))
    nfix = 0
    if opts["pfix"] is not None:
        pfix = np.array(opts["pfix"], dtype="d")
        nfix = len(pfix)
        logger.info(f"Constraining {nfix} fixed points..")
    return pfix, nfix


def _get_topology(dt):
    """Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    return dt.get_finite_vertices(), dt.get_finite_cells()


def _closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)
