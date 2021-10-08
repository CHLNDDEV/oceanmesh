import numpy as np

import oceanmesh


def test_del_exterior():
    # a mesh that has 1/5 of the total area disjoint.
    p = np.array(
        [
            [0, 0],
            [0, 0.33333],
            [0, 0.66667],
            [0, 1],
            [0.33333, 0],
            [0.33333, 0.33333],
            [0.33333, 0.66667],
            [0.33333, 1],
            [0.66667, 0],
            [0.66667, 0.33333],
            [0.66667, 0.66667],
            [0.66667, 1],
            [1, 0],
            [1, 0.33333],
            [1, 0.66667],
            [1, 1],
            [1.5, 1.5],
            [1.5, 1.625],
            [1.5, 1.75],
            [1.5, 1.875],
            [1.5, 2],
            [1.625, 1.5],
            [1.625, 1.625],
            [1.625, 1.75],
            [1.625, 1.875],
            [1.625, 2],
            [1.75, 1.5],
            [1.75, 1.625],
            [1.75, 1.75],
            [1.75, 1.875],
            [1.75, 2],
            [1.875, 1.5],
            [1.875, 1.625],
            [1.875, 1.75],
            [1.875, 1.875],
            [1.875, 2],
            [2, 1.5],
            [2, 1.625],
            [2, 1.75],
            [2, 1.875],
            [2, 2],
        ]
    )
    t = np.array(
        [
            [14, 10, 13],
            [5, 1, 4],
            [9, 5, 8],
            [0, 4, 1],
            [2, 1, 5],
            [6, 2, 5],
            [5, 4, 8],
            [7, 3, 6],
            [9, 6, 5],
            [3, 2, 6],
            [11, 7, 10],
            [11, 10, 14],
            [7, 6, 10],
            [15, 11, 14],
            [9, 8, 12],
            [10, 9, 13],
            [10, 6, 9],
            [9, 12, 13],
            [24, 19, 23],
            [18, 17, 22],
            [33, 32, 37],
            [34, 33, 38],
            [28, 23, 27],
            [21, 17, 16],
            [18, 23, 19],
            [21, 22, 17],
            [31, 36, 32],
            [26, 27, 22],
            [22, 21, 26],
            [23, 18, 22],
            [33, 34, 29],
            [19, 24, 20],
            [27, 23, 22],
            [29, 25, 24],
            [28, 24, 23],
            [25, 20, 24],
            [29, 30, 25],
            [29, 34, 30],
            [39, 40, 35],
            [39, 34, 38],
            [28, 27, 32],
            [35, 34, 39],
            [35, 30, 34],
            [37, 32, 36],
            [33, 28, 32],
            [29, 28, 33],
            [29, 24, 28],
            [27, 26, 31],
            [27, 31, 32],
            [38, 33, 37],
        ],
        dtype=int,
    )
    A1 = np.sum(oceanmesh.simp_vol(p, t))
    t2 = oceanmesh.delete_exterior_faces(p, t, 0.20)
    A2 = np.sum(oceanmesh.simp_vol(p, t2))
    assert (A1 - A2) == 0.25
