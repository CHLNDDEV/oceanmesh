import numpy
import skfmm

from .Grid import Grid
from inpoly import inpoly
from get_poly_edges import get_poly_edges

__all__ = ["signed_distance_function"]


def signed_distance_function(shoreline):
    """Takes a `shoreline` object containing segments represeting islands and mainland boundaries
    and calculates a signed distance function with it (assuming its a closed polygon)

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
    dis = skfmm.distance(phi, grid.grid_spacing)
    # now sign it
    edges = get_poly_edges(poly)
    inside, _ = inpoly(numpy.vstack((lon, lat)), poly, edges)
    grid.values = dis * inside
    grid.build_interpolant()
    return grid.eval
