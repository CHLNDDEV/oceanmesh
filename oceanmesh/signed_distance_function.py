import numpy
import scipy.spatial

from . import edges
from .inpoly import inpoly

__all__ = ["signed_distance_function", "Domain"]

nan = numpy.nan


class Domain:
    def __init__(self, bbox, domain):
        self.bbox = bbox
        self.domain = domain

    def eval(self, x):
        return self.domain(x)


def signed_distance_function(shoreline):
    """Takes a `shoreline` object containing segments representing islands and mainland boundaries
    and calculates a signed distance function with it (assuming the polygons are all closed).
    This function is queried every meshing iteration.

    Parameters
    ----------
    shoreline: a :class:`Shoreline` object
        The processed shapefile data from :class:`geodata`

    Returns
    -------
    domain: a :class:`Domain` object
        Contains a signed distance function with a bbox

    """
    print("Building a signed distance function...")
    poly = numpy.vstack((shoreline.inner, shoreline.mainland, shoreline.boubox))
    tree = scipy.spatial.cKDTree(poly[~numpy.isnan(poly[:, 0]), :])
    e = edges.get_poly_edges(poly)
    e_box = edges.get_poly_edges(shoreline.boubox)

    def func(x):
        # Initialize d with some positive number larger than geps
        dist = numpy.zeros(len(x)) + 1.0
        in_boubox, _ = inpoly(x, shoreline.boubox, e_box)
        # for points inside boubox, compute dist to shoreline
        if numpy.sum(in_boubox) > 0:
            in_outer, _ = inpoly(x[in_boubox == 1], poly, e)
            d, _ = tree.query(x[in_boubox == 1], k=1)
        # d is signed negative if inside and vice versa.
        dist[in_boubox == 1] = (-1) ** (in_outer) * d
        return dist

    return Domain(shoreline.bbox, func)
