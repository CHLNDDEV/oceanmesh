import numpy
import skfmm

from . import edges
from .grid import Grid
from .inpoly import inpoly

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
    qry = numpy.column_stack((lon.ravel(), lat.ravel()))

    inside, _ = inpoly(qry, poly, e)

    grid.values = dis * inside.reshape((dis.shape))
    grid.build_interpolant()
    return grid
