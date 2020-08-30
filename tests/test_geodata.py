import os

import pytest

from pyOceanMesh import DEM, Shoreline

fname = os.path.join(os.path.dirname(__file__), "GSHHS_l_L1.shp")
dfname = os.path.join(os.path.dirname(__file__), "galv_sub.nc")
tfname = os.path.join(os.path.dirname(__file__), "galv_sub.tif")


@pytest.mark.parametrize(
    "boxes_h0",
    [((166.0, 176.0, -48.0, -40.0), 1000.0), ((-74.0, -70.0, 35.0, 42.0), 50.0)],
)
def test_shoreline(boxes_h0):
    """Read in a shapefile at different scales h0
    shoreline and test you get the write output"""
    bbox, h0 = boxes_h0
    shp = Shoreline(fname, bbox, h0)
    assert len(shp.inner) > 0
    assert len(shp.mainland) > 0


@pytest.mark.parametrize(
    "files_bboxes",
    [
        (
            dfname,
            (-95.25, -95.2000000014, 28.9000000112, 29.099907413009998),
        ),
        (
            tfname,
            (-95.24, -95.20, 29.56252314079749, 29.80004628229499),
        ),
    ],
)
def test_geodata(files_bboxes):
    """Read in a subset of a DEM from netcdf/tif"""

    f, bbox = files_bboxes
    dem = DEM(f, bbox)
    assert isinstance(dem, DEM), "DEM class did not form"
