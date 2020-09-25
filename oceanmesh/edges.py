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


def get_boundary_edges(entities, dim=2):
    """Get the boundary edges of the mesh. Boundary edges only appear (dim-1) times

    Parameters
    ----------
    entities: array-like
        the mesh connectivity

    Returns
    -------
    boundary_edges: array-like
        the edges that make up the boundary of the mesh

    """
    edges = get_edges(entities, dim=dim)
    edges = np.sort(edges, axis=1)
    unq, cnt = np.unique(edges, axis=0, return_counts=True)
    boundary_edges = np.array([e for e, c in zip(unq, cnt) if c == (dim - 1)])
    return boundary_edges


def get_edges(entities, dim=2):
    """Get the undirected edges of mesh in no order (NB: are repeated)

    Parameters
    ----------
    entities: array-like
        the mesh connectivity
    dim: int, optional
        dimension of the mesh

    Returns
    -------
    edges: array-like
        the edges that make up the mesh

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
