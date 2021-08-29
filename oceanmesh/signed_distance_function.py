import numpy
import scipy.spatial
from inpoly import inpoly2

from . import edges

__all__ = ["multiscale_signed_distance_function", "signed_distance_function", "Domain"]

nan = numpy.nan


class Domain:
    def __init__(self, bbox, domain):
        self.bbox = bbox
        self.domain = domain

    def eval(self, x):
        return self.domain(x)


def multiscale_signed_distance_function(shorelines, verbose=True, flips=None):
    """Takes a list of `shoreline` objects and calculates a signed distance function from it.
        This functionis queried every meshing iteration.

    Parameters
    ----------
    shorelines: a list of :class:`Shoreline` objects
        Each item in the list is processed from :class:`geodata`
    flips: a list of booleans, optional
        Each boolean corresponds to the objects in shorelines

    Returns
    -------
    domain: a :class:`Domain` object
        A signed distance function representing the domain
    """
    if verbose:
        print("Building a multiscale signed distance function...")
    assert isinstance(shorelines, list), "shorelines is not a list"
    assert len(shorelines) > 1, "Use signed_distance_function instead"
    if flips is not None:
        assert isinstance(flips, list)
        assert len(flips) == len(shorelines)

    # build all SDF for each shoreline object
    sdfs = []
    for shoreline in shorelines:
        sdfs.apend(signed_distance_function(shoreline, verbose))

    def func(x):
        # query all the sdfs
        for sdf in sdfs:
            print("wip")
        return 0


def signed_distance_function(shoreline, verbose=True, flip=0):
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
    if verbose > 0:
        print("Building a signed distance function...")
    poly = numpy.vstack((shoreline.inner, shoreline.mainland, shoreline.boubox))
    tree = scipy.spatial.cKDTree(poly[~numpy.isnan(poly[:, 0]), :], balanced_tree=False)
    e = edges.get_poly_edges(poly)

    boubox = shoreline.boubox
    e_box = edges.get_poly_edges(boubox)

    def func(x):
        # Initialize d with some positive number larger than geps
        dist = numpy.zeros(len(x)) + 1.0
        # are points inside the boubox?
        in_boubox, _ = inpoly2(x, boubox, e_box)
        # are points inside the shoreline?
        in_shoreline, _ = inpoly2(x, poly, e)
        # compute dist to shoreline
        d, _ = tree.query(x, k=1)
        # d is signed negative if inside the
        # intersection of two areas and vice versa.
        cond = numpy.logical_and(in_shoreline, in_boubox)
        if flip:
            cond = ~cond
        dist = (-1) ** (cond) * d
        return dist

    return Domain(shoreline.bbox, func)
