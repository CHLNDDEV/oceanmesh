import time

import numpy as np
import scipy.sparse as spsparse

from .cpp.delaunay_class import DelaunayTriangulation as DT
from .fix_mesh import fix_mesh
from .grid import Grid
from .signed_distance_function import Domain

__all__ = ["generate_mesh"]


def silence(func):
    def wrapper(*args, **kwargs):
        None

    return wrapper


def talk(func):
    def wrapper(*args, **kwargs):
        func(*args, **kwargs)

    return wrapper


def _select_verbosity(opts):
    if opts["verbose"] == 0:
        return silence, silence
    elif opts["verbose"] == 1:
        return talk, silence
    elif opts["verbose"] > 1:
        return talk, talk

    else:
        raise ValueError("Unknown verbosity level")


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
            "edge_length",
            "bbox",
            "verbose",
        }:
            pass
        else:
            raise ValueError(
                "Option %s with parameter %s not recognized " % (key, kwargs[key])
            )


def generate_mesh(domain, edge_length, h0, **kwargs):
    r"""Generate a 2D triangular mesh using callbacks to a
        sizing function `edge_length` and signed distance function.

    Parameters
    ----------
    domain: A function object.
        A function that takes a point and returns the signed nearest distance to the domain boundary Ω.
    edge_length: A function object.
        A function that can evalulate a point and return a mesh size.
    h0: `float`
        The minimum element size in the domain.
    \**kwargs:
        See below

    :Keyword Arguments:
        * *bbox* (``tuple``) --
            Bounding box containing domain extents. REQUIRED IF NOT USING :class:`edge_length`
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
        * *verbose* (``int``) --
            Output to the screen `verbose` (default==1). If `verbose`==1 only start and end messages are

    Returns
    -------
    points: array-like
        vertex coordinates of mesh
    t: array-like
        mesh connectivity table.

    """
    opts = {
        "max_iter": 50,
        "seed": 0,
        "perform_checks": False,
        "pfix": None,
        "points": None,
        "verbose": 1,
    }
    opts.update(kwargs)
    _parse_kwargs(kwargs)

    verbosity1, verbosity2 = _select_verbosity(opts)

    @verbosity1
    def print_msg1(msg):
        print(msg, flush=True)

    @verbosity2
    def print_msg2(msg):
        print(msg, flush=True)

    fd, bbox = _unpack_domain(domain, opts)
    fh = _unpack_sizing(edge_length)

    if not isinstance(bbox, tuple):
        raise ValueError("`bbox` must be a tuple")

    dim = int(len(bbox) / 2)
    if dim != 2:
        raise ValueError("`dim` must be 2")

    bbox = np.array(bbox).reshape(-1, 2)

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

    pfix, nfix = _unpack_pfix(dim, opts)

    p = _generate_initial_points(h0, geps, bbox, fh, fd, pfix)

    N = p.shape[0]

    assert N > 0, "No vertices to mesh with!"

    count = 0
    print_msg1(
        "Commencing mesh generation with %d vertices." % (N),
    )

    while True:

        start = time.time()

        # (Re)-triangulation by the Delaunay algorithm
        dt = DT()
        dt.insert(p.ravel().tolist())

        # Get the current topology of the triangulation
        p, t = _get_topology(dt)

        # Remove points outside the domain
        t = _remove_triangles_outside(p, t, fd, geps)

        # Number of iterations reached, stop.
        if count == (max_iter - 1):
            p, t = _termination(p, t, verbose=opts["verbose"])
            break

        # Compute the forces on the bars
        Ftot = _compute_forces(p, t, fh, h0, L0mult)

        # Force = 0 at fixed points
        Ftot[:nfix] = 0

        # Update positions
        p += delta_t * Ftot

        # Bring outside points back to the boundary
        p = _project_points_back(p, fd, deps)

        # Show the user some progress so they know something is happening
        maxdp = delta_t * np.sqrt((Ftot ** 2).sum(1)).max()
        print_msg2(
            "Iteration #%d, max movement is %f, there are %d vertices and %d cells"
            % (count + 1, maxdp, len(p), len(t)),
        )

        count += 1

        end = time.time()
        print_msg2("     Elapsed wall-clock time %f : " % (end - start))

    return p, t


def _unpack_sizing(edge_length):
    if isinstance(edge_length, Grid):
        fh = edge_length.eval
    elif callable(edge_length):
        fh = edge_length
    else:
        raise ValueError(
            "`edge_length` must either be a function or a `edge_length` object"
        )
    return fh


def _unpack_domain(domain, opts):
    if isinstance(domain, Domain):
        bbox = domain.bbox
        fd = domain.eval
    elif callable(domain):
        bbox = opts["bbox"]
        fd = domain
    else:
        raise ValueError("`domain` must be a function or a :class:`geometry` object")
    return fd, bbox


def _termination(p, t, verbose):
    """Shut it down when reacing `max_iter`"""
    if verbose:
        print("Termination reached...maximum number of iterations reached.", flush=True)
    p, t, _ = fix_mesh(p, t, dim=2, delete_unused=True)
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
    N = p.shape[0]
    bars = _get_bars(t)
    barvec = p[bars[:, 0]] - p[bars[:, 1]]  # List of bar vectors
    L = np.sqrt((barvec ** 2).sum(1))  # L = Bar lengths
    L[L == 0] = np.finfo(float).eps
    hbars = fh(p[bars].sum(1) / 2)
    L0 = hbars * L0mult * ((L ** 2).sum() / (hbars ** 2).sum()) ** (1.0 / 2)
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

        dgrads = [(fd(p[ix] + _deps_vec(i)) - d[ix]) / deps for i in range(2)]
        dgrad2 = sum(dgrad ** 2 for dgrad in dgrads)
        dgrad2 = np.where(dgrad2 < deps, deps, dgrad2)
        p[ix] -= (d[ix] * np.vstack(dgrads) / dgrad2).T  # Project
    return p


def _generate_initial_points(h0, geps, bbox, fh, fd, pfix):
    """Create initial distribution in bounding box (equilateral triangles)"""
    p = np.mgrid[tuple(slice(min, max + h0, h0) for min, max in bbox)].astype(float)
    p = p.reshape(2, -1).T
    p = p[fd(p) < geps]  # Keep only d<0 points
    r0 = fh(p)
    r0m = np.min(r0[r0 > 0])
    return np.vstack(
        (
            pfix,
            p[np.random.rand(p.shape[0]) < r0m ** 2 / r0 ** 2],
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


def _get_topology(dt):
    """ Get points and entities from :clas:`CGAL:DelaunayTriangulation2/3` object"""
    return dt.get_finite_vertices(), dt.get_finite_cells()
