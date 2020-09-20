import os

from oceanmesh import Shoreline, signed_distance_function


def test_signed_distance_function():

    fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")

    bbox, h0 = (-74.0, -70.0, 40.0, 42.0), 500.0

    shp = Shoreline(fname, bbox, h0)

    domain = signed_distance_function(shp)

    domain.plot(vmin=-0.1, vmax=0.1)