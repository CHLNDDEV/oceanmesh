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


def test_shoreline_accepts_region_object():
    """Passing a Region object should auto-use its bbox and CRS."""
    region = Region((-74.0, -73.5, 40.0, 40.5), 4326)
    h0 = 0.01
    shp = Shoreline(fname, region, h0)
    # CRS should match the Region's CRS
    assert str(shp.crs.to_string()) == str(region.crs.to_string())
    assert len(shp.inner) >= 0  # construction succeeded


def test_shoreline_region_overrides_explicit_crs(caplog):
    """If both Region and explicit crs are provided, Region's CRS takes precedence and warns."""
    region = Region((-74.0, -73.5, 40.0, 40.5), 4326)
    h0 = 0.01
    caplog.clear()
    shp = Shoreline(fname, region, h0, crs=32610)
    assert "Region's CRS will take precedence" in " ".join(
        r.message for r in caplog.records
    )
    assert str(shp.crs.to_string()) == str(region.crs.to_string())


@pytest.mark.skipif(
    not os.path.exists(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "om_issue",
            "shoreline_for_om_32610_buffered.gpkg",
        )
    ),
    reason="UTM test data not available",
)
def test_shoreline_autodetects_projected_tuple_bbox(caplog):
    """Projected bbox without explicit CRS should trigger override log and adopt native projected CRS."""
    gpkg = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "om_issue",
        "shoreline_for_om_32610_buffered.gpkg",
    )
    import geopandas as gp

    region_gdf = gp.read_file(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "om_issue", "region.gpkg"
        )
    ).to_crs(32610)
    xmin, ymin, xmax, ymax = region_gdf.total_bounds
    bbox_utm = (xmin + 500, xmax - 500, ymin + 500, ymax - 500)
    h0 = 15.0
    import logging

    caplog.clear()
    caplog.set_level(logging.INFO, logger="oceanmesh.geodata")
    shp = Shoreline(gpkg, bbox_utm, h0)
    assert shp.crs.to_epsg() == 32610
    assert any("bbox looks projected" in r.message for r in caplog.records)


def test_shoreline_geographic_tuple_no_override(caplog):
    """Geographic bbox with default CRS should not trigger projected override log and remain EPSG:4326."""
    bbox_geo = (-75.0, -74.9, 40.0, 40.1)
    h0 = 0.01
    import logging

    caplog.clear()
    caplog.set_level(logging.INFO, logger="oceanmesh.geodata")
    shp = Shoreline(fname, bbox_geo, h0)  # default crs
    assert shp.crs.to_epsg() == 4326
    assert all("bbox looks projected" not in r.message for r in caplog.records)


def test_shoreline_error_message_guidance():
    """Deliberately mismatch CRS to ensure helpful error text is present."""
    bbox_utm = (500000.0, 501000.0, 4100000.0, 4101000.0)
    h0 = 100.0
    with pytest.raises(ValueError) as ei:
        _ = Shoreline(fname, bbox_utm, h0, crs=4326)
    msg = str(ei.value)
    assert "does not intersect" in msg
    assert "Region object" in msg or "crs=YOUR_CRS" in msg


@pytest.mark.parametrize(
    "files_bboxes",
    [
        (
            dfname,
            (-95.24, -95.21, 28.95, 29.00),
        ),
        (
            tfname,
            (-95.24, -95.21, 28.95, 29.00),
        ),
    ],
)
def test_geodata(files_bboxes):
    """Read in a subset of a DEM from netcdf/tif"""

    f, bbox = files_bboxes
    region = Region(bbox, 4326)
    dem = DEM(f, bbox=region, crs=region.crs)
    assert isinstance(dem, DEM), "DEM class did not form"
