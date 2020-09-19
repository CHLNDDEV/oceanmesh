import numpy
import skfmm

from .grid import Grid
from .inpoly import inpoly
from . import edges

__all__ = ["signed_distance_function"]


def signed_distance_function(shoreline):
    """Takes a `shoreline` object containing segments representing islands and mainland boundaries
    and calculates a signed distance function with it (assuming the polygons are all closed)

    Parameters
    ----------

    Returns
    -------

    """

    print("Building a signed distance function...")
    grid = Grid(bbox=shoreline.bbox, grid_spacing=shoreline.h0)
    phi = numpy.ones(shape=(grid.nx, grid.ny))
    lon, lat = grid.create_grid()
    poly = numpy.vstack((shoreline.inner, shoreline.mainland))
    indices = grid.find_indices(poly, lon, lat)
    phi[indices] = -1.0
    # call Fast Marching Method
    print("Calculating the distance...")
    dis = skfmm.distance(phi, grid.grid_spacing)
    # now sign it
    e = edges.get_poly_edges(poly)
    print("Signing the distance...")
    lon = numpy.reshape(lon, -1)
    lat = numpy.reshape(lat, -1)
    qry = numpy.vstack((lon, lat)).T
    print(len(qry), len(poly))
    import time

    t1 = time.time()
    inside, _ = inpoly(qry, poly, e)
    print(time.time() - t1)

    import matplotlib.pyplot as plt

    plt.pcolor(inside.reshape((dis.shape)).T)

    plt.show()

    grid.values = dis * inside.reshape((dis.shape))
    grid.build_interpolant()
    return grid.eval
