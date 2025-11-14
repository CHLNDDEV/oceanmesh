"""
Vendored from inpoly-python by Darren Engwirda
Source: https://github.com/dengwirda/inpoly-python
License: Custom (see LICENSE_INPOLY.txt)

Note: Vendored into oceanmesh to avoid external dependency build issues.
This file is a near-verbatim copy of the upstream pure-Python implementation.

Acceleration gating: if the environment variable PYTHON_INPOLY_ACCEL is set to
a truthy value, oceanmesh will attempt to import the optional compiled kernel
`oceanmesh._vendor.inpoly.inpoly_` for speed. Otherwise, the pure-Python
implementation is used. See the bottom of this file for details.
"""

import numpy as np
import os


def inpoly2(vert, node, edge=None, ftol=5.0e-14):
    """
      INPOLY2: compute "points-in-polygon" queries.

      STAT = INPOLY2(VERT, NODE, EDGE) returns the "inside/ou-
      tside" status for a set of vertices VERT and a polygon
      NODE, EDGE embedded in a two-dimensional plane. General
      non-convex and multiply-connected polygonal regions can
      be handled. VERT is an N-by-2 array of XY coordinates to
      be tested. STAT is an associated N-by-1 boolean array,
      with STAT[II] = TRUE if VERT[II, :] is an inside point.

      The polygonal region is defined as a piecewise-straight-
      line-graph, where NODE is an M-by-2 array of polygon ve-
      rtices and EDGE is a P-by-2 array of edge indexing. Each
      row in EDGE represents an edge of the polygon, such that
      NODE[EDGE[KK, 0], :] and NODE[EDGE[KK, 2], :] are the
      coordinates of the endpoints of the KK-TH edge. If the
      argument EDGE is omitted it assumed that the vertices in
      NODE are connected in ascending order.

      STAT, BNDS = INPOLY2(..., FTOL) also returns an N-by-1
      boolean array BNDS, with BNDS[II] = TRUE if VERT[II, :]
      lies "on" a boundary segment, where FTOL is a floating-
      point tolerance for boundary comparisons. By default,
      FTOL ~ EPS ^ 0.85.

      --------------------------------------------------------

      This algorithm is based on a "crossing-number" test,
      counting the number of times a line extending from each
      point past the right-most end of the polygon intersects
      with the polygonal boundary. Points with odd counts are
      "inside". A simple implementation requires that each
      edge intersection be checked for each point, leading to
      O(N*M) complexity...

      This implementation seeks to improve these bounds:

    * Sorting the query points by y-value and determining can-
      didate edge intersection sets via binary-search. Given a
      configuration with N test points, M edges and an average
      point-edge "overlap" of H, the overall complexity scales
      like O(M*H + M*LOG(N) + N*LOG(N)), where O(N*LOG(N)) operat-
      ions required for the set of binary-searches, and O(M*H)
      operations required for the intersection tests, where H
      is typically small on average, such that H << N.

    * Carefully checking points against the bounding-box asso-
      ciated with each polygon edge. This minimises the number
      of calls to the (relatively) expensive edge intersection
      test.

      Updated: 19 Dec, 2020

      Authors: Darren Engwirda, Keith Roberts

    """

    vert = np.asarray(vert, dtype=np.float64)
    node = np.asarray(node, dtype=np.float64)

    STAT = np.full(vert.shape[0], False, dtype=bool)
    BNDS = np.full(vert.shape[0], False, dtype=bool)

    if node.size == 0:
        return STAT, BNDS

    if edge is None:
        # ----------------------------------- set edges if not passed
        indx = np.arange(0, node.shape[0] - 1)

        edge = np.zeros((node.shape[0], 2), dtype=np.int32)

        edge[:-1, 0] = indx + 0
        edge[:-1, 1] = indx + 1
        edge[-1, 0] = node.shape[0] - 1

    else:
        edge = np.asarray(edge, dtype=np.int32)

    # ----------------------------------- prune points using bbox
    used = edge.ravel()
    xmin = np.nanmin(node[used, 0])
    xmax = np.nanmax(node[used, 0])
    ymin = np.nanmin(node[used, 1])
    ymax = np.nanmax(node[used, 1])
    xdel = xmax - xmin
    ydel = ymax - ymin

    lbar = (xdel + ydel) / 2.0

    veps = lbar * ftol

    mask = np.logical_and.reduce(
        (
            vert[:, 0] >= xmin - veps,
            vert[:, 1] >= ymin - veps,
            vert[:, 0] <= xmax + veps,
            vert[:, 1] <= ymax + veps,
        )
    )

    vert = vert[mask]

    if vert.size == 0:
        return STAT, BNDS

    # ------------------ flip to ensure y-axis is the `long` axis
    xdel = np.amax(vert[:, 0]) - np.amin(vert[:, 0])
    ydel = np.amax(vert[:, 1]) - np.amin(vert[:, 1])

    lbar = (xdel + ydel) / 2.0

    if xdel > ydel:
        vert = vert[:, (1, 0)]
        node = node[:, (1, 0)]

    # ----------------------------------- sort points via y-value
    swap = node[edge[:, 1], 1] < node[edge[:, 0], 1]
    temp = edge[swap]
    edge[swap, :] = temp[:, (1, 0)]

    # ----------------------------------- call crossing-no kernel
    stat, bnds = _inpoly(vert, node, edge, ftol, lbar)

    # ----------------------------------- unpack array reindexing
    STAT[mask] = stat
    BNDS[mask] = bnds

    return STAT, BNDS


def _inpoly(vert, node, edge, ftol, lbar):
    """
    _INPOLY: the local pycode version of the crossing-number
    test. Loop over edges; do a binary-search for the first
    vertex that intersects with the edge y-range; crossing-
    number comparisons; break when the local y-range is met.

    """

    feps = ftol * (lbar**+1)
    veps = ftol * (lbar**+1)

    stat = np.full(vert.shape[0], False, dtype=bool)
    bnds = np.full(vert.shape[0], False, dtype=bool)

    # ----------------------------------- compute y-range overlap
    ivec = np.argsort(vert[:, 1], kind="quicksort")

    XONE = node[edge[:, 0], 0]
    XTWO = node[edge[:, 1], 0]
    YONE = node[edge[:, 0], 1]
    YTWO = node[edge[:, 1], 1]

    XMIN = np.minimum(XONE, XTWO)
    XMAX = np.maximum(XONE, XTWO)

    XMIN = XMIN - veps
    XMAX = XMAX + veps
    YMIN = YONE - veps
    YMAX = YTWO + veps

    YDEL = YTWO - YONE
    XDEL = XTWO - XONE

    EDEL = np.abs(XDEL) + YDEL

    ione = np.searchsorted(vert[:, 1], YMIN, "left", sorter=ivec)
    itwo = np.searchsorted(vert[:, 1], YMAX, "right", sorter=ivec)

    # ----------------------------------- loop over polygon edges
    for epos in range(edge.shape[0]):

        xone = XONE[epos]
        xtwo = XTWO[epos]
        yone = YONE[epos]
        ytwo = YTWO[epos]
        xmin = XMIN[epos]
        xmax = XMAX[epos]
        edel = EDEL[epos]
        xdel = XDEL[epos]
        ydel = YDEL[epos]

        # ------------------------------- calc. edge-intersection
        for jpos in range(ione[epos], itwo[epos]):

            jvrt = ivec[jpos]

            if bnds[jvrt]:
                continue

            xpos = vert[jvrt, 0]
            ypos = vert[jvrt, 1]

            if xpos >= xmin:
                if xpos <= xmax:
                    # ------------------- compute crossing number
                    mul1 = ydel * (xpos - xone)
                    mul2 = xdel * (ypos - yone)

                    if feps * edel >= abs(mul2 - mul1):
                        # ------------------- BNDS -- approx. on edge
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == yone) and (xpos == xone):
                        # ------------------- BNDS -- match about ONE
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (ypos == ytwo) and (xpos == xtwo):
                        # ------------------- BNDS -- match about TWO
                        bnds[jvrt] = True
                        stat[jvrt] = True

                    elif (mul1 <= mul2) and (ypos >= yone) and (ypos < ytwo):
                        # ------------------- advance crossing number
                        stat[jvrt] = not stat[jvrt]

            elif (ypos >= yone) and (ypos < ytwo):
                # ----------------------- advance crossing number
                stat[jvrt] = not stat[jvrt]

    return stat, bnds


# Try to import the compiled Cython kernel for performance when enabled via
# environment variable. If not available or disabled, the pure-Python
# implementation above will be used automatically.
_COMPILED_KERNEL_AVAILABLE = False
_ACCEL_ENV = os.environ.get("PYTHON_INPOLY_ACCEL")
if _ACCEL_ENV and _ACCEL_ENV not in ("0", "false", "False", "no", "off"):
    try:
        from oceanmesh._vendor.inpoly.inpoly_ import _inpoly  # type: ignore  # noqa: F811

        _COMPILED_KERNEL_AVAILABLE = True
    except (ImportError, ModuleNotFoundError) as _ex:  # narrow exception set
        # Optional debug logging when import fails
        if os.environ.get("PYTHON_INPOLY_ACCEL_DEBUG"):
            print(
                f"[inpoly2] Compiled kernel import failed: {_ex}. "
                "Falling back to pure-Python implementation."
            )
