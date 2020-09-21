import numpy as np
from numba import jit


def inpoly(vert, node, edge=None, ftol=4.9485e-16):
    """A port of Darren Engwirda's `inpoly` routine into Python.

       Returns the "inside/outside" status for a set of vertices VERT and
       a polygon {NODE,EDGE}  embedded in a two-dimensional plane. General
       non-convex and multiply-connected polygonal regions can be handled.


    Parameters
    ----------
    vert: array-like
        VERT is an N-by-2 array of XY coordinates to query.

    node: array-like
        NODE is an M-by-2 array of polygon vertices.

    edge: array-like, optional
        EDGE is a P-by-2 array of edge indexing. Each
        row in EDGE represents an edge of the polygon, such that
        node(edge(KK,1),:) and node(edge(KK,2),:) are the coord-
        inates of the endpoints of the KK-TH edge. If the argum-
        ent EDGE is omitted it assumed that the vertices in NODE
        are connected in ascending order.

    ftol: float, otpional
        FTOL is a floating-point tolerance for boundary comparisons.
        By default, FTOL = EPS ^ 0.85.


    Returns:
    --------
    STAT: array-like
        STAT is an associated N-by-1 logical array,
        with STAT(II) = TRUE if VERT(II,:) is an interior point.

    BNDS: array-like
        N-by-1 logical array BNDS, with BNDS(II) = TRUE if VERT(II,:)
        lies "on" a boundary segment,


    Notes
    -----

    ALL CREDIT GOES TO THE ORIGINAL AUTHOR: Darren Engwirda

    Darren Engwirda : 2017 --
    Email           : de2363@columbia.edu
    Last updated    : 31/03/2020


    Author of port:

    Keith J. Roberts (2020)
    Email: krober@usp.br
    Date: 2020-13-09

    """

    nnod = len(node)
    nvrt = len(vert)

    # create edge if not supplied
    if edge is None:
        edge = np.vstack((np.arange(0, nnod - 1), np.arange(1, nnod))).T
        edge = np.concatenate((edge, [[nnod - 1, 0]]))

    STAT = np.zeros((nvrt), dtype=int)
    BNDS = np.zeros((nvrt), dtype=int)

    # prune points using bbox
    mask = (
        (vert[:, 0] >= np.nanmin(node[:, 0]))
        & (vert[:, 0] <= np.nanmax(node[:, 0]))
        & (vert[:, 1] >= np.nanmin(node[:, 1]))
        & (vert[:, 1] <= np.nanmax(node[:, 1]))
    )

    vert = vert[mask]

    # flip to ensure y-axis is the `long` axis
    vmin = np.amin(vert, axis=0)
    vmax = np.amax(vert, axis=0)
    ddxy = vmax - vmin

    lbar = np.sum(ddxy) / 2.0

    if ddxy[0] > ddxy[1]:
        vert = vert[:, [1, 0]]
        node = node[:, [1, 0]]

    # sort points via y-value
    swap = node[edge[:, 1], 1] < node[edge[:, 0], 1]

    tmp = edge[swap]
    edge[swap, :] = tmp[:, [1, 0]]

    ivec = np.argsort(vert[:, 1])
    vert = vert[ivec]

    t_stat, t_bnds = _inpoly(vert, node, edge, ftol, lbar)

    stat = np.ones(len(vert))
    bnds = np.ones(len(vert))
    stat[ivec] = t_stat
    bnds[ivec] = t_bnds

    STAT[mask] = stat
    BNDS[mask] = bnds

    return STAT, BNDS


@jit(nopython=True)
def _inpoly(vert, node, edge, ftol, lbar):

    feps = ftol * lbar ** 2
    veps = ftol * lbar ** 1

    _nvrt = len(vert)
    _nedge = len(edge)

    _stat = np.zeros((_nvrt))
    _bnds = np.zeros((_nvrt))

    # loop over polygon edges
    for epos in range(_nedge):

        inod = edge[epos, 0]
        jnod = edge[epos, 1]

        # calc. edge bounding-box
        yone = node[inod, 1]
        ytwo = node[jnod, 1]
        xone = node[inod, 0]
        xtwo = node[jnod, 0]

        xmin = np.minimum(xone, xtwo)
        xmax = np.maximum(xone, xtwo)

        xmax += veps

        ymin = yone - veps
        ymax = ytwo + veps

        ydel = ytwo - yone
        xdel = xtwo - xone

        # find top VERT[:,1]<YONE
        ilow = 0  # for python
        iupp = _nvrt

        while ilow < iupp - 1:  # binary search
            imid = int(ilow + np.floor((iupp - ilow) / 2))
            if vert[imid, 1] < ymin:
                ilow = imid
            else:
                iupp = imid

        if vert[ilow, 1] >= ymin:
            ilow -= 1

        # calc. edge-intersection
        for jpos in range(ilow + 1, _nvrt):

            if _bnds[jpos]:
                continue

            xpos = vert[jpos, 0]
            ypos = vert[jpos, 1]

            if ypos <= ymax:
                if xpos >= xmin:
                    if xpos <= xmax:
                        # compute crossing number
                        mul1 = ydel * (xpos - xone)
                        mul2 = xdel * (ypos - yone)
                        if feps >= np.abs(mul2 - mul1):
                            # BNDS -- approx. on edge
                            _bnds[jpos] = 1
                            _stat[jpos] = 1
                        elif (ypos == yone) & (xpos == xone):
                            # BNDS -- match about ONE
                            _bnds[jpos] = 1
                            _stat[jpos] = 1
                        elif (ypos == ytwo) & (xpos == xtwo):
                            # BNDS -- match about TWO
                            _bnds[jpos] = 1
                            _stat[jpos] = 1
                        elif mul1 < mul2:
                            if (ypos >= yone) & (ypos < ytwo):
                                # advance crossing number
                                _stat[jpos] = 1 - _stat[jpos]

                else:
                    if (ypos >= yone) & (ypos < ytwo):
                        # advance crossing number
                        _stat[jpos] = 1 - _stat[jpos]
            else:
                # done -- due to the sort
                break
    return _stat, _bnds
