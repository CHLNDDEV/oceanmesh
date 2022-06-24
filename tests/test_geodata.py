import os

import pytest

from oceanmesh import DEM, Region, Shoreline, edges

fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")
dfname = os.path.join(os.path.dirname(__file__), "galv_sub.nc")
tfname = os.path.join(os.path.dirname(__file__), "galv_sub.tif")


@pytest.mark.parametrize(
    "boxes_h0",
    [((166.0, 176.0, -48.0, -40.0), 0.01), ((-74.0, -70.0, 35.0, 42.0), 0.005)],
)
def test_shoreline(boxes_h0):
    """Read in a shapefile at different scales h0
    shoreline and test you get the write output"""
    bbox, h0 = boxes_h0
    region = Region(bbox, 4326)
    shp = Shoreline(fname, region.bbox, h0, crs=region.crs)
    assert len(shp.inner) > 0
    assert len(shp.mainland) > 0
    e = edges.get_poly_edges(shp.inner)
    edges.draw_edges(shp.inner, e)


@pytest.mark.parametrize(
    "files_bboxes",
    [
        (dfname, (-95.24, -95.21, 28.95, 29.00),),
        (tfname, (-95.24, -95.21, 28.95, 29.00),),
    ],
)
def test_geodata(files_bboxes):
    """Read in a subset of a DEM from netcdf/tif"""

    f, bbox = files_bboxes
    region = Region(bbox, 4326)
    dem = DEM(f, bbox=region.bbox, crs=region.crs)
    assert isinstance(dem, DEM), "DEM class did not form"
