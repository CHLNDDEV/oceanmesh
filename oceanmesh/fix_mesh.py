import warnings

import numpy as np


def simp_qual(p, t):
    """Simplex quality radius-to-edge ratio
    :param p: vertex coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :return: signed mesh quality: signed mesh quality (1.0 is perfect)
    :rtype: numpy.ndarray[`float` x 1]
    """
    assert p.ndim == 2 and t.ndim == 2 and p.shape[1] + 1 == t.shape[1]

    def length(p1):
        return np.sqrt((p1**2).sum(1))

    a = length(p[t[:, 1]] - p[t[:, 0]])
    b = length(p[t[:, 2]] - p[t[:, 0]])
    c = length(p[t[:, 2]] - p[t[:, 1]])
    # Suppress Runtime warnings here because we know that mult1/denom1 can be negative
    # as the mesh is being cleaned
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mult1 = (b + c - a) * (c + a - b) * (a + b - c) / (a + b + c)
        denom1 = np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
        r = 0.5 * mult1
        R = a * b * c / denom1
        return 2 * r / R


def fix_mesh(p, t, ptol=2e-13, dim=2, delete_unused=False):
    """Remove duplicated/unused vertices and entities and
       ensure orientation of entities is CCW.
    :param p: point coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :param ptol: point tolerance to detect duplicates
    :type ptol: `float`, optional
    :param dim: dimension of mesh
    :type dim: `int`, optional
    :param delete_unused: flag to delete disjoint vertices.
    :type delete_unused: `boolean`, optional
    :return: p: updated point coordinates of mesh
    :rtype: numpy.ndarray[`float` x dim]
    :return: t: updated mesh connectivity
    :rtype: numpy.ndarray[`int` x (dim+1)]
    """

    # duplicate vertices
    snap = (p.max(0) - p.min(0)).max() * ptol
    _, ix, jx = unique_rows(np.round(p / snap) * snap, True, True)

    p = p[ix]
    t = jx[t]

    # duplicate entities
    t = np.sort(t, axis=1)
    t = unique_rows(t)

    # delete disjoint vertices
    if delete_unused:
        pix, _, jx = np.unique(t, return_index=True, return_inverse=True)
        t = np.reshape(jx, (t.shape))
        p = p[pix, :]

    # entity orientation is CCW
    flip = simp_vol(p, t) < 0
    t[flip, :2] = t[flip, 1::-1]

    return p, t, jx


def unique_rows(A, return_index=False, return_inverse=False):
    """Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    :param  A: array of data
    :type A: numpy.ndarray[`int`/`float` x N]
    :param return_index: whether to return the indices of unique data
    :type return_index: `boolean`, optional
    :param return_inverse: whether to return the inverse mapping back to A from B.
    :type return_inverse: `boolean`, optional
    :return: B: array of data with duplicates removed
    :rtype: numpy.ndarray[`int`/`float` x N]
    :return: I: array of indices to unique data B.
    :rtype: numpy.ndarray[`int` x 1]
    :return: J: array of indices to A from B.
    :rtype: numpy.ndarray[`int` x 1]
    """
    A = np.require(A, requirements="C")
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((f"S{orig_dtype.itemsize * ncolumns}"))
    B, I, J = np.unique(A.view(dtype), return_index=True, return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order="C")

    # There must be a better way to do this:
    if return_index:
        if return_inverse:
            return B, I, J
        else:
            return B, I
    else:
        if return_inverse:
            return B, J
        else:
            return B


def simp_vol(p, t):
    """Signed volumes of the simplex elements in the mesh.
    :param p: point coordinates of mesh
    :type p: numpy.ndarray[`float` x dim]
    :param t: mesh connectivity
    :type t: numpy.ndarray[`int` x (dim + 1)]
    :return: volume: signed volume of entity/simplex.
    :rtype: numpy.ndarray[`float` x 1]
    """

    dim = p.shape[1]
    if dim == 1:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        return d01
    elif dim == 2:
        d01 = p[t[:, 1]] - p[t[:, 0]]
        d02 = p[t[:, 2]] - p[t[:, 0]]
        return (d01[:, 0] * d02[:, 1] - d01[:, 1] * d02[:, 0]) / 2
    elif dim == 3:
        d01 = p[t[:, 1], :] - p[t[:, 0], :]
        d02 = p[t[:, 2], :] - p[t[:, 0], :]
        d03 = p[t[:, 3], :] - p[t[:, 0], :]
        return np.einsum("ij,ij->i", np.cross(d01, d02), d03) / 6
    else:
        raise NotImplementedError
