import logging
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial
from inpoly import inpoly2

from . import Shoreline, edges

logger = logging.getLogger(__name__)

__all__ = [
    "multiscale_signed_distance_function",
    "signed_distance_function",
    "Domain",
    "Union",
    "Difference",
    "Intersection",
    "create_circle",
]

nan = np.nan


def create_circle(center, radius):
    """Create a circle centered on `center` and with
    radius `radius` in WGS84 degrees"""
    stepSize = 0.1
    positions = []
    t = 0
    while t < 2 * math.pi:
        positions.append(
            (radius * math.cos(t) + center[0], radius * math.sin(t) + center[1])
        )
        t += stepSize

    return np.array(positions)


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


def _plot(geo, filename=None, samples=100000):
    p = _generate_samples(geo.bbox, 2, N=samples)
    d = geo.eval(p)
    ix = np.logical_and(d > -0.0001, d < 0.0001)

    fig = plt.figure()
    ax = fig.add_subplot(111)  # , projection="3d")
    im = ax.scatter(p[ix, 0], p[ix, 1], c=d[ix], marker=".", s=5.0)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    plt.title("Approximate 0-level set")
    fig.colorbar(im, ax=ax)
    im.set_clim(-0.001, 0.001)
    ax.set_aspect("auto")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


class Domain:
    def __init__(self, bbox, func, covering=None):
        self.bbox = bbox
        self.domain = func
        self.covering = covering

    def eval(self, x):
        return self.domain(x)

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


# Note to self: these primitive operations are inexact.
# Recall: https://www.iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
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


def signed_distance_function(shoreline):
    """Takes a :class:`Shoreline` object containing linear segments representing meshing boundaries
    and calculates a signed distance function with it under the assumption that all polygons are closed.
    The returned function `func` becomes a bound method of the :class:`Domain` and is queried during
    mesh generation several times per iteration.

    Parameters
    ----------
    shoreline: a :class:`Shoreline` object
        The processed shapefile data from :class:`Geodata`

    Returns
    -------
    domain: a :class:`Domain` object
        Contains a signed distance function along with an extent `bbox`

    """
    logger.info("Building a signed distance function...")

    assert isinstance(shoreline, Shoreline), "shoreline is not a Shoreline object"
    poly = np.vstack((shoreline.inner, shoreline.boubox))
    tree = scipy.spatial.cKDTree(
        poly[~np.isnan(poly[:, 0]), :], balanced_tree=False, leafsize=50
    )
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
        try:
            d, _ = tree.query(x, k=1, workers=-1)
        except (Exception,):
            d, _ = tree.query(x, k=1, n_jobs=-1)
        # d is signed negative if inside the
        # intersection of two areas and vice versa.
        cond = np.logical_and(in_shoreline, in_boubox)
        dist = (-1) ** (cond) * d
        return dist

    return Domain(shoreline.bbox, func, covering=boubox)


def _create_boubox(bbox):
    """Create a bounding box from domain extents `bbox`. Path orientation will be CCW."""
    xmin, xmax, ymin, ymax = bbox
    return np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
            [xmin, ymin],
        ],
        dtype=float,
    )


def multiscale_signed_distance_function(signed_distance_functions):
    """Takes a list of :class:`Domain` objects and calculates a signed distance
        function from each one that represents a multiscale meshing domain.

    Parameters
    ----------
    signed_distance_functions: a list of `signed_distance_function` objects

    Returns
    -------
    union: a :class:`Union` object
        The union of the `signed_distance_functions`
    nests: a list of :class:`Difference` containing objects
        Nested domains are set differenced from their parent domains.
    """
    logger.info("Building a multiscale signed distance function...")

    msg = "`signed_distance_functions` is not a list"
    assert isinstance(signed_distance_functions, list), msg
    assert len(signed_distance_functions) > 1, "Use `signed_distance_function` instead"
    msg = "list does not contain all `signed_distance_function`"
    for sdf in signed_distance_functions:
        assert isinstance(sdf, Domain), msg

    # calculate the boolean/set difference from the base sdf and subsequent nests
    nests = []
    for i, sdf in enumerate(signed_distance_functions):
        nests.append(Difference([sdf, *signed_distance_functions[i + 1 :]]))

    union = Union(nests)

    return union, nests
