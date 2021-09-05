import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from inpoly import inpoly2

from . import Shoreline, edges

__all__ = [
    "multiscale_signed_distance_function",
    "signed_distance_function",
    "Domain",
    "Union",
    "Difference",
    "Intersection",
]

nan = np.nan


def _generate_samples(bbox, dim, N):
    N = int(N)
    points = []
    _xrange = (bbox[0] - 0.01, bbox[1] + 0.01)
    _yrange = (bbox[2] - 0.01, bbox[3] + 0.01)
    if dim == 2:
        points.append(
            [
                (
                    random.uniform(*_xrange),
                    random.uniform(*_yrange),
                )
                for i in range(N)
            ]
        )
    elif dim == 3:
        _zrange = (bbox[4] - 0.01, bbox[5] + 0.01)
        points.append(
            [
                (
                    random.uniform(*_xrange),
                    random.uniform(*_yrange),
                    random.uniform(*_zrange),
                )
                for i in range(N)
            ]
        )
    points = np.asarray(points)
    points = points.reshape(-1, dim)
    return points


def _plot(geo, filename=None, samples=10000):
    p = _generate_samples(geo.bbox, 2, N=samples)
    d = geo.eval(p)
    ix = np.logical_and(d > -0.01, d < 0.01)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    im = ax.scatter(p[ix, 0], p[ix, 1], p[ix, 0] * 0.0, c=d[ix], marker=".", s=5.0)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    plt.title("Approximate 0-level set")
    fig.colorbar(im, ax=ax)
    im.set_clim(-0.1, 0.1)
    ax.set_aspect("auto")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


class Domain:
    def __init__(self, bbox, domain):
        self.bbox = bbox
        self.domain = domain

    def eval(self, *args, **kwargs):
        return self.domain(*args, **kwargs)

    def plot(self, filename=None, samples=10000):
        _plot(self, filename=None, samples=samples)


def _compute_bbox(domains):
    bbox = (
        min(d.bbox[0] for d in domains),
        max(d.bbox[1] for d in domains),
        min(d.bbox[2] for d in domains),
        max(d.bbox[3] for d in domains),
    )
    return bbox


class Union(Domain):
    def __init__(self, domains):
        bbox = _compute_bbox(domains)
        super().__init__(bbox, domains)

    def eval(self, x):
        d = [d.eval(x) for d in self.domain]
        return np.minimum.reduce(d)


class Intersection(Domain):
    def __init__(self, domains):
        bbox = _compute_bbox(domains)
        super().__init__(bbox, domains)

    def eval(self, x):
        d = [d.eval(x) for d in self.domain]
        return np.maximum.reduce(d)


class Difference(Domain):
    def __init__(self, domains):
        bbox = _compute_bbox(domains)
        super().__init__(bbox, domains)

    def eval(self, x):
        return np.maximum.reduce(
            [-d.eval(x) if n > 0 else d.eval(x) for n, d in enumerate(self.domain)]
        )


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
    assert isinstance(shoreline, Shoreline), "shoreline is not a Shoreline object"
    poly = np.vstack((shoreline.inner, shoreline.boubox))
    tree = scipy.spatial.cKDTree(poly[~np.isnan(poly[:, 0]), :], balanced_tree=False)
    e = edges.get_poly_edges(poly)

    boubox = shoreline.boubox
    e_box = edges.get_poly_edges(boubox)

    def func(x):
        # Initialize d with some positive number larger than geps
        dist = np.zeros(len(x)) + 1.0
        # are points inside the boubox?
        in_boubox, _ = inpoly2(x, boubox, e_box)
        # are points inside the shoreline?
        in_shoreline, _ = inpoly2(x, poly, e)
        # compute dist to shoreline
        d, _ = tree.query(x, k=1)
        # d is signed negative if inside the
        # intersection of two areas and vice versa.
        cond = np.logical_and(in_shoreline, in_boubox)
        if flip:
            cond = ~cond
        dist = (-1) ** (cond) * d
        return dist

    return Domain(shoreline.bbox, func)


def multiscale_signed_distance_function(signed_distance_functions, verbose=True):
    """Takes a list of :class:`signed_distance_function` objects and calculates a signed distance
        function from each one.

    Parameters
    ----------
    signed_distance_functions: a list of `signed_distance_function` objects

    Returns
    -------
    union: a :class:`Union` object
        The union of all signed distance functions
    nests: a list of `Difference` objects
        All inner domains are differenced from the domain above.

    """
    if verbose:
        print("Building a multiscale signed distance function...")
    assert isinstance(
        signed_distance_functions, list
    ), "`signed_distance_function` is not a list"
    assert len(signed_distance_functions) > 1, "Use `signed_distance_function` instead"

    union = Union(signed_distance_functions)

    nests = []
    for ix1, sdf_base in enumerate(signed_distance_functions):
        nests.append(Difference([sdf_base, *signed_distance_functions[ix1 + 1 :]]))

    return union, nests
