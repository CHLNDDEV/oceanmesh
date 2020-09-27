import copy

import numpy as np

from fix_mesh import fix_mesh
from geometry import simp_vol
import edges

__all__ = ["make_mesh_boundaries_traversable"]


def _arg_sortrows(arr):
    """Before a multi column sort like MATLAB's sortrows"""
    i = arr[:, 1].argsort()  # First sort doesn't need to be stable.
    j = arr[i, 0].argsort(kind="mergesort")
    return i[j]


def _cell_to_cell(t):
    """Cell to cell connectivity table.
        Cell `i` is connected to cells `ctoc[ix[i]:ix[i+1]]`
        By connected, I mean shares a mutual edge.

    Parameters
    ----------
    t: array-like
        Mesh connectivity table.

    Returns
    -------
    ctoc: array-like
        Cell numbers connected to cells.
    ix: array-like
        indices into `ctoc`

    """
    nt = len(t)
    t = np.sort(t, axis=1)
    # NB: we use order="F" to reshape because the `np.tile` command below
    e = t[:, [[0, 1], [0, 2], [1, 2]]].reshape((nt * 3, 2), order="F")
    trinum = np.tile(np.arange(nt), 3)
    j = _arg_sortrows(e)
    e = e[j, :]
    trinum = trinum[j]
    k = np.argwhere(~np.diff(e, axis=0).any(axis=1))
    ctoc = np.concatenate((trinum[k], trinum[k + 1]), axis=1)
    ctoc = np.append(ctoc, np.fliplr(ctoc), axis=0)
    ctoc = ctoc[np.argsort(ctoc[:, 0]), :]
    idx = np.argwhere(np.diff(ctoc[:, 0])) + 1
    idx = np.insert(idx, 0, 0)
    idx = np.append(idx, len(ctoc))
    return ctoc, idx


def make_mesh_boundaries_traversable(points, cells, dj_cutoff=0.05):
    """
    A mesh described by points and cells is  "cleaned" and returned.
    Alternates between checking interior and exterior portions
    of the mesh exhaustively until convergence is obtained, defined as:
    Having no vertices connected to more than two boundary edges.

    Parameters
    ----------
    points: array-like
        The vertices of the "uncleaned" mesh.
    cells: array-like
        The "cleaned" mesh connectivity.
    dj_cutoff: float
        A decimal percentage used to decide whether to keep or remove
        disconnected portions of the meshing domain.


    Returns
    -------
    points: array-like
        The vertices of the "cleaned" mesh.

    cells: array-like
        The "cleaned" mesh connectivity.

    Notes
    -----

    Interior Check: Deletes cells that are within the interior of the
    mesh so that no nodes are connected to more than 2 boundary edges. For
    example, a split could become very thin in a middle portion so that you
    a node is connected to two cells but four boundary edges, in a
    bow-tie type formation. This code will delete one of those connecting
    cells to ensure the spit is continous and only two boundary edges
    are connected to that node. In the case of a choice between cells to
    delete, the one with the lowest quality is chosen.

    Exterior Check: Finds small disjoint portions of the mesh and removes
    them using a breadth-first search. The individual disjoint portions are
    removed based on `dj_cutoff`.

    """

    boundary_edges, boundary_points = _external_topology(points, cells)

    # NB: when this inequality is not met, the mesh boundary '
    # is valid.
    while len(boundary_edges) > len(boundary_points):

        cells = _delete_exterior_cells(points, cells, dj_cutoff)
        points, cells = fix_mesh(points, cells)

        cells = _delete_interior_cells(points, cells)
        points, cells = fix_mesh(points, cells)

        boundary_edges, boundary_points = _external_topology(points, cells)


def _external_topology(points, cells):
    """Get edges and points that make up the boundary of the mesh"""
    boundary_edges = edges.get_boundary(points, cells)
    boundary_points = points[np.unique(boundary_edges.reshape(-1))]
    return boundary_edges, boundary_points


def _delete_exterior_cells(points, cells, dj_cutoff):
    """Deletes portions of the mesh that are "outside" or not
    connected to the majority which represent a fractional
    area less than `dj_cutoff`.
    """
    t1 = copy.copy(cells)
    t = []
    # Calculate the total area of the patch
    A = np.sum(simp_vol(points, cells))
    An = A
    while An / A > dj_cutoff:
        # Perform the Breadth-First-Search to get `nflag`
        nflag = _breadth_first_search(points, t1)

        # Get new triangulation and its area
        t2 = t1[nflag == 1, :]
        An = np.sum(simp_vol(points, t2))

        # If large enough, retain this component
        if An / A > dj_cutoff:
            t = np.append(t, t2, axis=0)

        # Delete where nflag == 1 from tmp t1 mesh
        t1 = np.delete(t1, nflag == 1, axis=0)

        # Calculate the remaining area
        An = np.sum(simp_vol(points, t1))

    p_cleaner, t_cleaner = fix_mesh(points, t)

    return p_cleaner, t_cleaner


def _delete_interior_cells(points, cells, dj_cutoff):
    return 0


def _breadth_first_search(points, cells):
    """Breadth-First-Search across the triangulation"""

    nt = len(cells)
    EToS = np.random.randint(0, 1, nt)
    # Get cell-to-cell connectivity.
    ctoc, ix = _cell_to_cell(cells)

    # Traverse grid deleting elements outside.
    ic = np.zeros(np.ceil(np.sqrt(nt) * 2))
    ic0 = np.zeros(np.ceil(np.sqrt(nt) * 2))
    nflag = np.zeros(nt)

    ic[0] = EToS
    icc = 1

    # Using BFS loop over until convergence is reached (i.e., we
    # obtained a connected region).
    while icc:
        ic0[:icc] = ic[:icc]
        icc0 = icc
        icc = 0
        for nn in range(icc0):
            i = ic0[nn]
            # Flag the current cell as OK
            nflag[i] = 1
            # Search neighboring elements
            nb = ctoc[ix[i] : ix[i + 1]]
            # Flag connected neighbors as OK
            for nnb in range(len(nb)):
                if not nflag[nb[nnb]]:
                    icc += 1
                    ic[icc] = nb[nnb]
                    nflag[nb[nnb]] = 1
    return nflag
