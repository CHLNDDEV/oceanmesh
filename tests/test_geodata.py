import os

import pytest

from pyOceanMesh import Shoreline

fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")


@pytest.mark.parametrize(
    "boxes_h0",
    [((166.0, 176.0, -48.0, -40.0), 1000.0), ((-74.0, -70.0, 35.0, 42.0), 50)],
)
def test_shoreline(boxes_h0):
    """ Read in a shapefile at different scales h0
        shoreline and test you get the write output"""
    bbox, h0 = boxes_h0
    shp = Shoreline(shp=fname, bbox=bbox, h0=h0)
    assert len(shp.inner) > 0
    assert len(shp.mainland) > 0


@pytest.mark.parametrize("bboxes_h0", [((), 50.0)])
def test_geodata(bboxes_h0):
    """Read in several subsets of a DEM"""
