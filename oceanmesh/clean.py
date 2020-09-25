import numpy as np

import edges

__all__ = ["make_mesh_boundaries_traversable"]


def make_mesh_boundaries_traversable(points, cells, dj_cutoff=0.05):
    """
    A mesh described by points and cells is  "cleaned" and returned.
    Alternates between checking interior and exterior portions
    of the graph exhaustively until convergence is obtained, defined as:
    Having no nodes connected to more than 2 boundary edges.

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

    Interior Check: Deletes elements that are within the interior of the
    mesh so that no nodes are connected to more than 2 boundary edges. For
    example, a split could become very thin in a middle portion so that you
    a node is connected to two elements but four boundary edges, in a
    bow-tie type formation. This code will delete one of those connecting
    elements to ensure the spit is continous and only two boundary edges
    are connected to that node. In the case of a choice between elements to
    delete, the one with the lowest quality is chosen.

    Exterior Check: Finds small disjoint portions of the graph and removes
    them using a breadth-first search. The individual disjoint portions are
    removed based on `dj_cutoff`.

    """
    boundary_edges = edges.get_boundary(points, cells)
    boundary_points = points[np.unique(boundary_edges.reshape(-1))]
