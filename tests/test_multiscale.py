import os
import numpy as np
import oceanmesh as om


fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")


def test_multiscale_sizing_function():

    bbox1, min_edge_length1 = (-75.000, -70.001, 40.0001, 41.9000), 1e3

    bbox2, min_edge_length2 = (
        np.array(
            [[-74.25, 40.5], [-73.75, 40.55], [-73.75, 41], [-74, 41], [-74.25, 40.5]]
        ),
        500,
    )

    bbox3, min_edge_length3 = (
        np.array(
            [
                [-73.9799, 40.6539],
                [-74.1117, 40.6634],
                [-74.0790, 40.7118],
                [-73.9983, 40.7328],
                [-73.9339, 40.7296],
                [-73.9093, 40.6908],
                [-73.9492, 40.6598],
            ]
        ),
        250,
    )

    shore1 = om.Shoreline(fname, bbox1, min_edge_length1)
    edge_length1 = om.distance_sizing_function(shore1, max_edge_length=5e3)

    shore2 = om.Shoreline(fname, bbox2, min_edge_length2)
    edge_length2 = om.distance_sizing_function(shore2, max_edge_length=1e3)

    shore3 = om.Shoreline(fname, bbox3, min_edge_length3)
    edge_length3 = om.distance_sizing_function(shore3, max_edge_length=500)

    el_org = [edge_length1, edge_length2, edge_length3]
    el_new, minimum_edge_length = om.create_multiscale(el_org)

    assert minimum_edge_length * 111e3 == 250