import numpy as np

import oceanmesh


def test_del_interior():
    # a mesh that has two elements that delete interior should remove
    p = np.array(
        [
            [0, 0],
            [0.5000, 0.5000],
            [1.0000, 0],
            [-0.5000, 0.5000],
            [1.5000, 0.5000],
            [0, 0.7500],
            [0.5000, 1.2500],
            [1.0000, 0.7500],
            [-0.5000, 1.2500],
            [1.5000, 1.2500],
            [0, 0.7500],
            [0.5000, 0.5000],
            [-0.5000, 0.5000],
        ]
    )

    t = np.array(
        [
            [1, 2, 3],
            [4, 1, 2],
            [5, 2, 3],
            [6, 7, 8],
            [9, 6, 7],
            [10, 7, 8],
            [4, 6, 2],
            [2, 8, 5],
        ],
        dtype=int,
    )

    t -= 1

    cells, deleted = oceanmesh.delete_interior_faces(p, t, verbose=0)
    assert np.allclose(deleted, [6, 7])
