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
        return np.sqrt((p1 ** 2).sum(1))

    a = length(p[t[:, 1]] - p[t[:, 0]])
    b = length(p[t[:, 2]] - p[t[:, 0]])
    c = length(p[t[:, 2]] - p[t[:, 1]])
    r = 0.5 * np.sqrt((b + c - a) * (c + a - b) * (a + b - c) / (a + b + c))
    R = a * b * c / np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
    return 2 * r / R


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
