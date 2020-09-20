import time

import numpy as np
import scipy.sparse as spsparse

from .cpp.delaunay_class import DelaunayTriangulation as DT2
from .fix_mesh import fix_mesh
from .grid import Grid
from .signed_distance_function import Domain

__all__ = ["generate_mesh"]

opts = {
    "nscreen": 1,
    "max_iter": 50,
    "seed": 0,
    "perform_checks": False,
    "pfix": None,
    "points": None,
}


def _parse_kwargs(kwargs):
    for key in kwargs:
        if key in {
            "nscreen",
            "max_iter",
            "seed",
            "perform_checks",
            "pfix",
            "points",
            "domain",
            "cell_size",
            "bbox",
        }:
            pass
        else:
            raise ValueError(
                "Option %s with parameter %s not recognized " % (key, kwargs[key])
            )


def generate_mesh(domain, cell_size, h0, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to a
        sizing function `cell_size` and signed distance function.

    Parameters
    ----------
    domain: A function object.
        A function that takes a point and returns the signed nearest distance to the domain boundary Ω.
    cell_size: A function object.
        A function that can evalulate a point and return a mesh size.
    h0: `float`
        The minimum element size in the domain.
    \**kwargs:
        See below

    :Keyword Arguments:
        * *bbox* (``tuple``) --
            Bounding box containing domain extents. REQUIRED IF NOT USING :class:`cell_size`
        * *nscreen* (``float``) --
            Output to the screen `nscreen` timestep. (default==1)
        * *max_iter* (``float``) --
            Maximum number of meshing iterations. (default==50)
        * *seed* (``float`` or ``int``) --
            Psuedo-random seed to initialize meshing points. (default==0)
        * *perform_checks* (`boolean`) --
            Whether or not to perform mesh linting/mesh cleanup. (default==False)
        * *pfix* (`array-like`) --
            An array of points to constrain in the mesh. (default==None)
        * *axis* (`int`) --
            The axis to decompose the mesh (1,2, or 3). (default==1)

    Returns
    -------
    points: array-like
        vertex coordinates of mesh
    t: array-like
        mesh connectivity table.

    """
    # check call was correct
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    # unpack domain
    fd, bbox = _unpack_domain(domain)
    fh = _unpack_sizing(cell_size)

    if not isinstance(bbox, tuple):
        raise ValueError("`bbox` must be a tuple")

    # check bbox shape
    dim = int(len(bbox) / 2)
    if dim != 2:
        raise ValueError("`dim` must be 2")

    bbox = np.array(bbox).reshape(-1, 2)

    # check h0
    if h0 < 0:
        raise ValueError("`h0` must be > 0")

    if opts["max_iter"] < 0:
        raise ValueError("`max_iter` must be > 0")
    max_iter = opts["max_iter"]

    np.random.seed(opts["seed"])

    L0mult = 1 + 0.4 / 2 ** (dim - 1)
    delta_t = 0.1
    geps = 1e-1 * h0
    deps = np.sqrt(np.finfo(np.double).eps) * h0

    DT = _select_cgal_dim()

    pfix, nfix = _unpack_pfix(dim, opts)

    p = _generate_initial_points(h0, geps, dim, bbox, fh, fd, pfix)

    N = p.shape[0]

    assert N > 0, "No vertices to mesh with!"

    count = 0
    print(
        "Commencing mesh generation with %d vertices." % (N),
        flush=True,
    )

    nscreen = opts["nscreen"]
    while True:

        start = time.time()

        # (Re)-triangulation by the Delaunay algorithm
        dt = DT()
        dt.insert(p.flatten().tolist())

        # Get the current topology of the triangulation
        p, t = _get_topology(dt)

        # Remove points outside the domain
        t = _remove_triangles_outside(p, t, fd, geps)

        # Compute the forces on the bars
        Ftot = _compute_forces(p, t, fh, h0, L0mult)

        # Force = 0 at fixed points
        Ftot[:nfix] = 0

        # Update positions
        p += delta_t * Ftot

        # Bring outside points back to the boundary
        p = _project_points_back(p, fd, deps)

        # Number of iterations reached, stop.
        if count == (max_iter - 1):
            p, t = _termination(p, t)
            break

        # Show the user some progress so they know something is happening
        if count % nscreen == 0:
            maxdp = delta_t * np.sqrt((Ftot ** 2).sum(1)).max()
            _display_progress(p, t, count, nscreen, maxdp)

        count += 1

        end = time.time()
        if count % nscreen == 0:
            print("     Elapsed wall-clock time %f : " % (end - start), flush=True)

    return p, t


def _unpack_sizing(cell_size):
    if isinstance(cell_size, Grid):
        fh = cell_size.eval
    elif callable(cell_size):
        fh = cell_size
    else:
        raise ValueError(
            "`cell_size` must either be a function or a `cell_size` object"
        )
    return fh


def _unpack_domain(domain):
    if isinstance(domain, Domain):
        bbox = domain.bbox
        fd = domain.eval
    elif callable(domain):
        bbox = opts["bbox"]
        fd = domain
    else:
        raise ValueError("`domain` must be a function or a :class:`geometry` object")
    return fd, bbox


def _display_progress(p, t, count, nscreen, maxdp):
    """print progress"""
    print(
        "Iteration #%d, max movement is %f, there are %d vertices and %d cells"
        % (count + 1, maxdp, len(p), len(t)),
        flush=True,
    )


def _termination(p, t):
    """Shut it down when reacing `max_iter`"""
    print("Termination reached...maximum number of iterations reached.", flush=True)
    dim = p.shape[1]
    p, t, _ = fix_mesh(p, t, dim=dim, delete_unused=True)
    return p, t


def _get_bars(t):
    """Describe each bar by a unique pair of nodes"""
    bars = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    return _unique_rows(np.ascontiguousarray(bars, dtype=np.uint32))


def _unique_rows(ar):
    ar_row_view = ar.view("|S%d" % (ar.itemsize * ar.shape[1]))
    _, unique_row_indices = np.unique(ar_row_view, return_index=True)
    ar_out = ar[unique_row_indices]
    return ar_out


def _compute_forces(p, t, fh, h0, L0mult):
    """Compute the forces on each edge based on the sizing function"""
    dim = p.shape[1]
    N = p.shape[0]
    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec ** 2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    hbars = fh(p[bars].sum(1) / 2)
    L0 = hbars * L0mult * ((L ** dim).sum() / (hbars ** dim).sum()) ** (1.0 / dim)
    F = L0 - L
    F[F < 0] = 0  # Bar forces (scalars)
    Fvec = (
        F[:, None] / L[:, None].dot(np.ones((1, dim))) * barvec
    )  # Bar forces (x,y components)

    Ftot = _dense(
        bars[:, [0] * dim + [1] * dim],
        np.repeat([list(range(dim)) * 2], len(F), axis=0),
        np.hstack((Fvec, -Fvec)),
        shape=(N, dim),
    )
    return Ftot


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
    dim = p.shape[1]
    pmid = p[t].sum(1) / (dim + 1)  # Compute centroids
    return t[fd(pmid) < -geps]  # Keep interior triangles


def _project_points_back(p, fd, deps):
    """Project points outsidt the domain back within"""
    dim = p.shape[1]

    d = fd(p)
    ix = d > 0  # Find points outside (d>0)
    if ix.any():

        def _deps_vec(i):
            a = [0] * dim
            a[i] = deps
            return a

        dgrads = [(fd(p[ix] + _deps_vec(i)) - d[ix]) / deps for i in range(dim)]
        dgrad2 = sum(dgrad ** 2 for dgrad in dgrads)
        dgrad2 = np.where(dgrad2 < deps, deps, dgrad2)
        p[ix] -= (d[ix] * np.vstack(dgrads) / dgrad2).T  # Project
    return p


def _generate_initial_points(h0, geps, dim, bbox, fh, fd, pfix):
    """Create initial distribution in bounding box (equilateral triangles)"""
    p = np.mgrid[tuple(slice(min, max + h0, h0) for min, max in bbox)].astype(float)
    p = p.reshape(dim, -1).T
    p = p[fd(p) < geps]  # Keep only d<0 points
    r0 = fh(p)
    r0m = np.min(r0[r0 > 0])
    return np.vstack(
        (
            pfix,
            p[np.random.rand(p.shape[0]) < r0m ** dim / r0 ** dim],
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
        print(
            "Constraining " + str(nfix) + " fixed points..",
            flush=True,
        )
    return pfix, nfix


def _select_cgal_dim():
    """Select back-end CGAL Delaunay call"""
    return DT2


def _get_topology(dt):
    """ Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    return dt.get_finite_vertices(), dt.get_finite_cells()
