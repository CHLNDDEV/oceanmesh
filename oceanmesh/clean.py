import copy

import numpy as np

from .fix_mesh import fix_mesh, simp_vol, simp_qual
from . import edges

__all__ = [
    "make_mesh_boundaries_traversable",
    "delete_interior_cells",
    "delete_exterior_cells",
]


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


def _vertex_to_cell(vertices, cells):
    """Determine which elements are connected to which vertices.

     Parameters
    ----------
    vertices: array-like
        Vertices of the mesh.
    t: array-like
        Mesh connectivity table.

    Returns
    -------
    vtoc: array-like
        cell numbers connected to vertices.
    ix: array-like
        indices into `vtoc`

    """
    num_cells = len(cells)

    ext = np.tile(np.arange(0, num_cells), (3, 1)).reshape(-1, order="F")
    ve = np.reshape(cells, (-1,))
    ve = np.vstack((ve, ext)).T
    ve = ve[ve[:, 0].argsort(), :]

    idx = np.insert(np.diff(ve[:, 0]), 0, 0)
    vtoc_pointer = np.argwhere(idx)
    vtoc_pointer = np.insert(vtoc_pointer, 0, 0)
    vtoc_pointer = np.append(vtoc_pointer, num_cells * 3)

    vtoc = ve[:, 1]

    return vtoc, vtoc_pointer


def make_mesh_boundaries_traversable(vertices, cells, dj_cutoff=0.05):
    """
    A mesh described by vertices and cells is  "cleaned" and returned.
    Alternates between checking "interior" and "exterior" portions
    of the mesh until convergence is obtained. Convergence is defined as:
    having no vertices connected to more than two boundary edges.

    Parameters
    ----------
    vertices: array-like
        The vertices of the "uncleaned" mesh.
    cells: array-like
        The "cleaned" mesh connectivity.
    dj_cutoff: float
        A decimal percentage used to decide whether to keep or remove
        disconnected portions of the meshing domain.


    Returns
    -------
    vertices: array-like
        The vertices of the "cleaned" mesh.

    cells: array-like
        The "cleaned" mesh connectivity.

    Notes
    -----

    Interior Check: Deletes cells that are within the interior of the
    mesh so that no vertices are connected to more than two boundary edges. For
    example, a barrier island could become very thin in a middle portion so that you
    have a vertex connected to two cells but four boundary edges, in a
    bow-tie type formation.

    This code will delete one of those connecting
    cells to ensure the spit is `clean` in the sense that two boundary edges
    are connected to that vertex. In the case of a choice between cells to
    delete, the one with the lowest quality is chosen.

    Exterior Check: Finds small disjoint portions of the mesh and removes
    them using a breadth-first search. The individual disjoint portions are
    removed based on `dj_cutoff` which is a decimal representing a fractional
    threshold component of the total mesh.

    """

    boundary_edges, boundary_vertices = _external_topology(vertices, cells)

    # NB: when this inequality is not met, the mesh boundary '
    # is valid and non-manifold
    while len(boundary_edges) > len(boundary_vertices):

        cells = delete_exterior_cells(vertices, cells, dj_cutoff)
        vertices, cells = fix_mesh(vertices, cells)

        cells = delete_interior_cells(vertices, cells)
        vertices, cells = fix_mesh(vertices, cells)

        boundary_edges, boundary_vertices = _external_topology(vertices, cells)


def _external_topology(vertices, cells):
    """Get edges and vertices that make up the boundary of the mesh"""
    boundary_edges = edges.get_boundary_edges(cells)
    boundary_vertices = vertices[np.unique(boundary_edges.reshape(-1))]
    return boundary_edges, boundary_vertices


def delete_exterior_cells(vertices, cells, dj_cutoff):
    """Deletes portions of the mesh that are "outside" or not
    connected to the majority which represent a fractional
    area less than `dj_cutoff`.
    """
    vertices, cells, _ = fix_mesh(vertices, cells, delete_unused=True)
    t1 = copy.copy(cells)
    t = []
    # Calculate the total area of the patch
    A = np.sum(simp_vol(vertices, cells))
    An = A
    while An / A > dj_cutoff:
        # Perform the Breadth-First-Search to get `nflag`
        nflag = _breadth_first_search(vertices, t1)

        # Get new triangulation and its area
        t2 = t1[nflag == 1, :]
        An = np.sum(simp_vol(vertices, t2))

        # If large enough, retain this component
        if An / A > dj_cutoff:
            t = np.append(t, t2, axis=0)

        # Delete where nflag == 1 from tmp t1 mesh
        t1 = np.delete(t1, nflag == 1, axis=0)
        print(
            "ACCEPTED: Deleting {}  cells outside the main mesh".format(np.sum(nflag))
        )

        # Calculate the remaining area
        An = np.sum(simp_vol(vertices, t1))

        vertices, t1, _ = fix_mesh(vertices, t1, delete_unused=True)

    p_cleaner, t_cleaner, _ = fix_mesh(vertices, t, delete_unused=True)

    return p_cleaner, t_cleaner


def delete_interior_cells(vertices, cells):
    """Delete interior cells that have vertices with more than
    two vertices declared as boundary vertices
    """
    # Get updated boundary topology
    boundary_edges, boundary_vertices = _external_topology(vertices, cells)
    etbv = boundary_edges.reshape(-1)
    # Count how many edges a vertex appears in.
    uebtv, count = np.unique(etbv, return_counts=True)
    # Get the cells connected to the vertices
    vtoc, nne = _vertex_to_cell(vertices, cells)
    # Vertices which appear more than twice (implying they are shared by
    # more than two boundary edges)
    del_cell_idx = []
    for ix in uebtv[count > 2]:
        conn_cells = vtoc[nne[ix] : nne[ix + 1]]
        del_cell = []
        for conn_cell in conn_cells:
            II = etbv == cells[conn_cell, 0]
            JJ = etbv == cells[conn_cell, 1]
            KK = etbv == cells[conn_cell, 2]
            if np.any(II) and np.any(JJ) and np.any(KK):
                del_cell.append(conn_cell)

        if len(del_cell) == 1:
            del_cell_idx.append(del_cell)
        elif len(del_cell) > 1:
            # Delete worst quality qualifying cell.
            qual = simp_qual(vertices, cells[del_cell])
            idx = np.argmin(qual)
            del_cell_idx.append(del_cell[idx])
        else:
            # No connected cells have all vertices on boundary edge so we
            # select the worst quality connecting cell.
            qual = simp_qual(vertices, cells[conn_cells])
            idx = np.argmin(qual)
            del_cell_idx.append(conn_cell(idx))

    print("ACCEPTED: Deleting {}  cells inside the main mesh".format(len(del_cell_idx)))
    cells = np.delete(cells, del_cell_idx, 0)

    return cells, del_cell_idx


def _breadth_first_search(points, cells):
    """Breadth-First-Search (BFS) across the triangulation"""

    nt = len(cells)

    # select a random cell
    selected = np.random.randint(0, nt, 1)

    # Get cell-to-cell connectivity.
    ctoc, ix = _cell_to_cell(cells)

    nflag = np.zeros(nt)

    searching = True

    visited = []
    visited.append(*selected)

    # Traverse through connected mesh
    while searching:
        searching = False
        for c in visited:
            # Flag the current cell as visited
            nflag[c] = 1
            # Search connected cells
            neis = ctoc[ix[c] : ix[c + 1], 1]
            # Flag connected cells as visited
            for nei in neis:
                if nflag[nei] == 0:
                    nflag[nei] = 1
                    visited.append(nei)
                    searching = True
    return nflag


# def _breadth_first_search(vertices, cells):
#    """Breadth-First-Search (BFS) across the triangulation"""
#
#    nt = len(cells)
#    EToS = np.random.randint(0, nt, 1)
#    ctoc, ix = _cell_to_cell(cells)
#
#    # temporary arrays
#    sz = int(np.ceil(np.sqrt(nt) * 2))
#    ic = np.zeros(sz, dtype=int)
#    ic0 = np.zeros(sz, dtype=int)
#    nflag = np.zeros(nt)
#
#    ic[0] = EToS
#    icc = 1
#
#    while icc:
#        ic0[:icc] = ic[:icc]
#        icc = 0
#        for curr in ic0:
#            nflag[curr] = 1
#            neis = ctoc[ix[curr] : ix[curr + 1], 1]
#            for nei in neis:
#                if nflag[nei] == 0:
#                    nflag[nei] = 1
#                    ic[icc] = nei
#                    icc += 1
#    return nflag
