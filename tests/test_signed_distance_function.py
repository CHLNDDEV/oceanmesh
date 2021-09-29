import os

from oceanmesh import Domain, Region, Shoreline, signed_distance_function


def test_signed_distance_function():

    fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")

    region = Region((-74.0, -70.0, 40.0, 42.0), 4326)
    h0 = 0.005

    shp = Shoreline(fname, region.bbox, h0, crs=region.crs)

    domain = signed_distance_function(shp)

    assert isinstance(domain, Domain)
