import pytest

from pyOceanMesh import Shoreline

fname = "GSHHS_l_L1.shp"


@pytest.mark.parametrize(
    "boxes_h0",
    [((166.0, 176.0, -48.0, -40.0), 1000.0), ((-74.0, -70.0, 35.0, 42.0), 50)],
)
def test_shoreline(boxes_h0):
    """ Read in a shapefile at different scales h0
        shoreline and test you get the write output"""
    bbox, h0 = boxes_h0
    shp = Shoreline(shp=fname, bbox=bbox, h0=h0)
