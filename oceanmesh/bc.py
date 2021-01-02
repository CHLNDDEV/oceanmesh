import numpy as np

from . import edges

__all__ = ["label_island_bcs"]


def label_island_bcs(vertices, faces, ibtype=21):
    """Labels the boundaries of internal loops ('islands') for an
    ADCIRC mesh
    """
    boundary_loops = edges.get_winded_boundary_loops(faces)
    nbou = len(boundary_loops[1:])
    nvell = np.zeros((1, nbou))
    max_len = [max(len(loop) for loop in boundary_loops[1:])]
    nbvv = np.zeros((max_len, nbou))
    nvel = 0
    for nbou, loop in enumerate(boundary_loops[1:]):
        vso = vertices[loop]
        nvell[nbou] = len(vso)
        nvel += nvell[nbou]
        nbvv[: nvell[nbou], nbou] = loop
    return nvell, nvel, nbvv
