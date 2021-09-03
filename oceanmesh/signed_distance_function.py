import numpy
import scipy.spatial
from inpoly import inpoly2

from . import Shoreline, edges

__all__ = [
    "multiscale_signed_distance_function",
    "signed_distance_function",
    "Domain",
    "MultiscaleDomain",
]

nan = numpy.nan


class Domain:
    def __init__(self, bbox, domain):
        self.bbox = bbox
        self.domain = domain

    def eval(self, *args, **kwargs):
        return self.domain(*args, **kwargs)


class MultiscaleDomain(Domain):
    def __init__(self, bboxes, domains):
        super().__init__(bboxes, domains)


def signed_distance_function(shoreline, verbose=True, flip=0):
    """Takes a `shoreline` object containing segments representing islands and mainland boundaries
    and calculates a signed distance function with it (assuming the polygons are all closed).
    This function is queried every meshing iteration.

    Parameters
    ----------
    shoreline: a :class:`Shoreline` object
        The processed shapefile data from :class:`geodata`
    verbose:
    flip:
    return_inside:

    Returns
    -------
    domain: a :class:`Domain` object
        Contains a signed distance function with a bbox

    """
    if verbose:
        print("Building a signed distance function...")
    assert isinstance(shoreline, Shoreline), "shoreline is not a `Shoreline` object"
    poly = numpy.vstack((shoreline.inner, shoreline.boubox))
    tree = scipy.spatial.cKDTree(poly[~numpy.isnan(poly[:, 0]), :], balanced_tree=False)
    e = edges.get_poly_edges(poly)

    boubox = shoreline.boubox
    e_box = edges.get_poly_edges(boubox)

    def func(x, box_vec=None):
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
    assert len(shorelines) > 1, "Use `signed_distance_function` instead"
    if flips is not None:
        assert isinstance(flips, list) and len(flips) == len(shorelines)

    # build all SDF for each shoreline object
    _sdfs = []
    for shoreline in shorelines:
        _sdfs.append(signed_distance_function(shoreline, verbose=False))

    def func(x, box_vec=None):

        if box_vec is None:
            box_vec = [*range(len(shorelines))]

        # initialize to large value
        sdist = numpy.zeros(len(x)) + 1.0

        # query the sdfs based on box_vec
        for box_num in box_vec:
            _shoreline = shorelines[box_num]
            # first determine the array `inside`
            if box_num > 0:
                _boubox = _shoreline.boubox
                e_box = edges.get_poly_edges(_boubox)
                inside, _ = inpoly2(x, _boubox, e_box)
            else:
                # by design all the points are inside
                inside = numpy.ones(len(x), dtype=bool)

            # all the nested domains are `outside`.
            for box_num2 in range(box_num + 1, len(shorelines)):
                _boubox = shorelines[box_num2].boubox
                e_box = edges.get_poly_edges(_boubox)
                inside2, _ = inpoly2(x, _boubox, e_box)
                inside[inside2] = False

            # for the points inside the box, calculate the nearest distance to
            if numpy.sum(inside) > 0:
                d_l = _sdfs[box_num].eval(x[inside, :])

                sdist[inside] = d_l

        return sdist

    return Domain([shoreline.bbox for shoreline in shorelines], func)
