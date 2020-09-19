import numpy as np
import matplotlib.pyplot as plt
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
