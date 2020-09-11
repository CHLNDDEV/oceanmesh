import os

from oceanmesh import Shoreline, distance_sizing_function


def test_edgefx():
    fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")

    bbox1 = (-75.0, -70.0, 38.0, 42.0)
    shore1 = Shoreline(fname, bbox1, 1000.0)

    dis1 = distance_sizing_function(shore1)

    bbox2 = (-74.0, -73.0, 40.0, 41.0)
    shore2 = Shoreline(fname, bbox2, 100.0)

    dis2 = distance_sizing_function(shore2)

    dis3 = dis2.project(dis1)

    ax = dis3.plot(hold=True)
    shore1.plot(ax)

    dis4 = dis1.project(dis2)
    ax = dis4.plot(hold=False)
