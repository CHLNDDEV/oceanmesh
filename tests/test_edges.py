import numpy as np

from oceanmesh import edges

nan = np.nan


def test_edges():
    poly = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [nan, nan],
            [0.2, 0.2],
            [0.4, 0.2],
            [0.3, 0.3],
            [0.2, 0.2],
            [nan, nan],
        ]
    )

    e = edges.get_poly_edges(poly)
    edges.draw_edges(poly, e)
