import os

import oceanmesh as om
from oceanmesh import Shoreline, distance_sizing_function

dfname = os.path.join(os.path.dirname(__file__), "galv_sub.nc")
fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")


def test_edgefx():

    bbox1 = (-75.0, -70.0, 38.0, 42.0)
    shore1 = Shoreline(fname, bbox1, 1000.0)
    shore1.plot()

    dis1 = distance_sizing_function(shore1)

    bbox2 = (-74.0, -73.0, 40.0, 41.0)
    shore2 = Shoreline(fname, bbox2, 100.0)

    dis2 = distance_sizing_function(shore2)
    dis2.extrapolate = False

    dis3 = dis2.interpolate_to(dis1)

    ax = dis3.plot(hold=True)
    shore1.plot(ax)

    dis4 = dis1.interpolate_to(dis2)
    ax = dis4.plot(hold=False)


def test_edgefx_elevation_bounds():
    bbox = (-95.24, -95.21, 28.95, 29.00)

    dem = om.DEM(dfname, bbox)

    sho = om.Shoreline(fname, bbox, 1e2)
    sho.plot()

    edge_length = om.distance_sizing_function(sho)

    bounds = [[2000.0, 3000.0, -10, -5], [1000.0, 1500.0, -5, -1]]
    edge_length = om.enforce_mesh_size_bounds_elevation(edge_length, dem, bounds)
    edge_length.plot()

def test_edgefx_medial_axis():

    bbox, min_edge_length = (-75.000, -70.001, 40.0001, 41.9000), 1e3

    shoreline = om.Shoreline(fname, bbox, min_edge_length)
    sdf = om.signed_distance_function(shoreline)

    # Visualize the medial points
    edge_length = om.feature_sizing_function(shoreline, sdf, max_edge_length=5e3, plot=True)
    ax = edge_length.plot(
        xlabel="longitude (WGS84 degrees)",
        ylabel="latitude (WGS84 degrees)",
        title="Feature sizing function",
        cbarlabel="mesh size (degrees)",
        hold=True,
        xlim=[-74.3, -73.8],
        ylim=[40.3, 40.8],
    )
    shoreline.plot(ax=ax)
