import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

nan = np.nan


def get_poly_edges(poly):
    """Given a winded polygon represented as a set of ascending line segments
    with separated features indicated by nans, this function calculates
    the edges of the polygon such that each edge indexes the start and end
    coordinates of each line segment of the polygon.

    Parameters
    ----------
    poly: array-like, float
        A 2D array of point coordinates with features sepearated by NaNs

    Returns
    -------
    edges: array-like, int
        A 2D array of integers containing indexes into the `poly` array.

    """
    ix = np.argwhere(np.isnan(poly[:, 0]))
    ix = np.insert(ix, 0, 0)

    edges = []
    for s in range(len(ix) - 1):
        col1 = np.arange(ix[s], ix[s + 1] - 1)
        col2 = np.arange(ix[s] + 1, ix[s + 1])
        tmp = np.vstack((col1, col2)).T
        tmp = np.append(tmp, [[ix[s + 1] - 1, ix[s] + 1]], axis=0)
        edges.append(tmp)
    return np.concatenate(edges, axis=0)


def draw_edges(poly, edges):
    """Visualizes the polygon as a bunch of line segments

    Parameters
    ----------
    poly: array-like, float
        A 2D array of point coordinates with features sepearated by NaNs.
    edges: array-like, int
        A 2D array of integers indexing into the `poly` array.

    Returns
    -------
    None

    """
    lines = []
    for edge in edges:
        lines.append([poly[edge[0]], poly[edge[1]]])
    lc = mc.LineCollection(lines, linewidths=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    plt.show()


def unique_row_view(data):
    """https://github.com/numpy/numpy/issues/11136"""
    b = np.ascontiguousarray(data).view(
        np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
    )
    u, cnts = np.unique(b, return_counts=True)
    u = u.view(data.dtype).reshape(-1, data.shape[1])
    return u, cnts


def get_edges(entities, dim=2):
    """Get the undirected edges of mesh in no order (NB: are repeated)

    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param dim: dimension of the mesh
    :type dim: `int`, optional

    :return: edges: the edges that make up the mesh
    :rtype: numpy.ndarray[`int`x 2]
    """

    num_entities = len(entities)
    entities = np.array(entities)
    if dim == 2:
        edges = entities[:, [[0, 1], [0, 2], [1, 2]]]
        edges = edges.reshape((num_entities * 3, 2))
    elif dim == 3:
        edges = entities[:, [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]]
        edges = edges.reshape((num_entities * 6, 2))
    return edges


def get_boundary_edges(entities, dim=2):
    """Get the boundary edges of the mesh. Boundary edges only appear (dim-1) times

    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param dim: dimension of the mesh
    :type dim: `int`, optional

    :return: boundary_edges: the edges that make up the boundary of the mesh
    :rtype: numpy.ndarray[`int` x 2]
    """
    edges = get_edges(entities, dim=dim)
    edges = np.sort(edges, axis=1)
    unq, cnt = unique_row_view(edges)
    boundary_edges = np.array([e for e, c in zip(unq, cnt) if c == (dim - 1)])
    return boundary_edges


def get_winded_boundary_edges(entities, vFirst=None):
    """Order the boundary edges of the mesh in a winding fashion

    :param entities: the mesh connectivity
    :type entities: numpy.ndarray[`int` x (dim+1)]
    :param vFirst: vertex index of any edge element to trace boundary along
    :type vFirst: `int`

    :return: boundary_edges: the edges that make up the boundary of the mesh in a winding order
    :rtype: numpy.ndarray[`int` x 2]
    """

    boundary_edges = get_boundary_edges(entities)
    _bedges = boundary_edges.copy()

    choice = 0
    if vFirst is not None:
        choice = next((i for i, j in enumerate(_bedges) if any(vFirst == j)), 0)

    isVisited = np.zeros((len(_bedges)))
    ordering = np.array([choice])
    isVisited[choice] = 1

    vStart, vNext = _bedges[choice, :]
    while True:
        locs = np.column_stack(np.where(_bedges == vNext))
        rows = locs[:, 0]
        choices = [row for row in rows if isVisited[row] == 0]
        if len(choices) == 0:
            break
        choice = choices[0]
        ordering = np.append(ordering, [choice])
        isVisited[choice] = 1
        nextEdge = _bedges[choice, :]
        tmp = [v for v in nextEdge if v != vNext]
        vNext = tmp[0]
    boundary_edges = boundary_edges[ordering, :]
    return boundary_edges
