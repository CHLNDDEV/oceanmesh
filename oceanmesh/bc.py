import numpy as np

from . import edges

__all__ = ["label_island_bcs"]


def label_island_bcs(vertices, faces, ibtype=21):
    """Labels the boundaries of internal loops ('islands') for an
    ADCIRC mesh
    """
    boundary_loops = edges.get_winded_boundary_loops(faces)
    nbou = len(boundary_loops[1:])
    nvell = np.zeros(nbou, dtype=int)
    max_len = max(len(loop[:-1].ravel()) for loop in boundary_loops[1:])
    nbvv = np.zeros((max_len, nbou), dtype=int)
    nvel = 0
    for nbou, loop in enumerate(boundary_loops[1:]):
        nvell[nbou] = len(loop[:-1].ravel())
        nvel += nvell[nbou]
        nbvv[: nvell[nbou], nbou] = loop[:-1].ravel()
    return nvel, nvell, nbvv
