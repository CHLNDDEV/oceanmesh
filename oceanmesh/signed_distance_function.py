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
    poly1 = numpy.vstack((shoreline.inner, shoreline.mainland))
    tree = scipy.spatial.cKDTree(poly1)

    boubox = _create_boubox(shoreline.bbox)
    poly2 = numpy.vstack((poly1, boubox))

    e = edges.get_poly_edges(poly2)

    def func(x):
        dist, _ = tree.query(x, k=1)
        inside, _ = inpoly(x, poly2, e)
        dist[inside == 1] *= -1.0
        return dist

    return Domain(shoreline.bbox, func)


def _create_boubox(bbox):
    """Create a bounding box from domain extents `bbox`."""
    xmin, xmax, ymin, ymax = bbox
    return [
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
        [xmin, ymin],
        [nan, nan],
    ]
