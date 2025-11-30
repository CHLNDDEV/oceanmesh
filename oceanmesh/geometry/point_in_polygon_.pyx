# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libc.math cimport fabs, isfinite

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t


def inpoly2_fast(np.ndarray[DTYPE_t, ndim=2] vert,
                 np.ndarray[DTYPE_t, ndim=2] node,
                 np.ndarray[ITYPE_t, ndim=2] edge,
                 double ftol):
    """Cython-accelerated ray-casting implementation of inpoly2.

    Parameters
    ----------
    vert : (M, 2) float64 ndarray
        Query points.
    node : (N, 2) float64 ndarray
        Polygon vertices.
    edge : (K, 2) int32 ndarray
        Edge indices into ``node``.
    ftol : float
        Distance tolerance for boundary detection.

    Returns
    -------
    stat : (M,) bool ndarray
        True where point is inside the polygon or on its boundary.
    bnds : (M,) bool ndarray
        True where point is classified as boundary.
    """

    cdef Py_ssize_t n_vert = vert.shape[0]
    cdef Py_ssize_t n_edge = edge.shape[0]

    cdef np.ndarray[np.uint8_t, ndim=1] stat_arr = np.zeros(n_vert, dtype=np.uint8)
    cdef np.ndarray[np.uint8_t, ndim=1] bnd_arr = np.zeros(n_vert, dtype=np.uint8)

    cdef DTYPE_t[:, ::1] v = vert
    cdef DTYPE_t[:, ::1] n = node
    cdef ITYPE_t[:, ::1] e = edge

    cdef Py_ssize_t i, j
    cdef ITYPE_t i0, i1
    cdef double x0, y0, x1, y1

    # Precompute edge vertical ranges.
    cdef np.ndarray[DTYPE_t, ndim=1] eymin = np.empty(n_edge, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] eymax = np.empty(n_edge, dtype=np.float64)
    cdef DTYPE_t[::1] eymin_view = eymin
    cdef DTYPE_t[::1] eymax_view = eymax

    for j in range(n_edge):
        i0 = e[j, 0]
        i1 = e[j, 1]
        y0 = n[i0, 1]
        y1 = n[i1, 1]
        if y0 < y1:
            eymin_view[j] = y0
            eymax_view[j] = y1
        else:
            eymin_view[j] = y1
            eymax_view[j] = y0

    cdef double px, py
    cdef double dx, dy, t, projx, projy, dist
    cdef double xints
    cdef bint inside, on_bnd

    for i in range(n_vert):
        px = v[i, 0]
        py = v[i, 1]

        # Skip NaNs early; caller is expected to mask these, but be robust.
        if not (isfinite(px) and isfinite(py)):
            continue

        inside = 0
        on_bnd = 0

        for j in range(n_edge):
            # Vertical range quick check.
            if py < eymin_view[j] - ftol or py > eymax_view[j] + ftol:
                continue

            i0 = e[j, 0]
            i1 = e[j, 1]
            x0 = n[i0, 0]
            y0 = n[i0, 1]
            x1 = n[i1, 0]
            y1 = n[i1, 1]

            dx = x1 - x0
            dy = y1 - y0

            # Boundary check: distance to segment <= ftol and projection within segment.
            if dx != 0.0 or dy != 0.0:
                t = ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy)
                if t >= 0.0 and t <= 1.0:
                    projx = x0 + t * dx
                    projy = y0 + t * dy
                    dist = fabs(px - projx) + fabs(py - projy)
                    if dist <= ftol:
                        on_bnd = 1
                        break

            # Ray casting: count crossings of horizontal ray to the right.
            if ((y0 <= py < y1) or (y1 <= py < y0)):
                # Avoid division by zero on vertical segments via dy check above.
                xints = x0 + (py - y0) * (dx / (dy if dy != 0.0 else 1e-16))
                if xints > px:
                    inside = not inside

        if on_bnd:
            bnd_arr[i] = 1
            stat_arr[i] = 1
        elif inside:
            stat_arr[i] = 1

    stat = stat_arr.astype(bool)
    bnds = bnd_arr.astype(bool)
    return stat, bnds
