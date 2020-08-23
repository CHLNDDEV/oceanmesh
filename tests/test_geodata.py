import os

import pytest


from pyOceanMesh import Shoreline, DEM

fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")
dfname = os.path.join(os.path.dirname(__file__), "galv_sub.nc")


@pytest.mark.parametrize(
    "boxes_h0",
    [((166.0, 176.0, -48.0, -40.0), 1000.0), ((-74.0, -70.0, 35.0, 42.0), 50.0)],
)
def test_shoreline(boxes_h0):
    """ Read in a shapefile at different scales h0
        shoreline and test you get the write output"""
    bbox, h0 = boxes_h0
    shp = Shoreline(fname, bbox, h0)
    assert len(shp.inner) > 0
    assert len(shp.mainland) > 0


@pytest.mark.parametrize("bboxes_h0", [((-95.22, -95.21, 29.02, 29.098), 10.0)])
def test_geodata(bboxes_h0):
    """Read in a subset of a DEM"""
    bbox, h0 = bboxes_h0
    dem = DEM(dfname, bbox)
    assert isinstance(dem, DEM), "DEM class did not form"
