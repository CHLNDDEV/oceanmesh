import os

import pytest
import oceanmesh as om

fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")
fdem = os.path.join(os.path.dirname(__file__), "../datasets/EastCoast.nc")


def generate_mesh(name, signed_distance, edge_length, max_iterations=50):
    points, cells = om.generate_mesh(
        signed_distance, edge_length, max_iter=max_iterations
    )
    # Makes sure the vertices of each triangle are arranged in an anti-clockwise manner
    points, cells, jx = om.fix_mesh(points, cells)
    # mesh_plot(points, cells, "1 Original Mesh")

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    # mesh_plot(points, cells, "2 Degenerate Elements Removed")

    # Removes faces connected by a single face
    points, cells = om.delete_faces_connected_to_one_face(points, cells)
    # mesh_plot(points, cells, "3 Deleted Faces Connected to One Face")

    # remove low quality boundary elements less than 15%
    points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
    # mesh_plot(points, cells, "4 Deleted Boundary Faces")

    # apply a Laplacian smoother
    P, T = om.laplacian2(points, cells)  # Final poost-processed mesh
    mesh_plot(P, T, f"Post-processed mesh with {name} edge function")

    return P, T


def mesh_plot(points, cells, plot_title=""):
    import matplotlib.pyplot as pt

    fig, ax = fig, ax = pt.subplots()
    ax.set_xlabel("longitude (EPSG:4326 degrees)")
    ax.set_ylabel("latitude (EPSG:4326 degrees)")
    ax.set_title(plot_title)

    X, Y = points.T
    ax.plot(X, Y, "bx", markersize=1)
    ax.triplot(X, Y, cells, linewidth=0.2, color="red")
    ax.set_aspect("equal")
    pt.show()


# @pytest.mark.skip(reason="not implemented yet")
def test_bathymetric_gradient_function():
    EPSG = 4326  # EPSG:4326 or WGS84
    bbox = (-74.4, -73.4, 40.2, 41.2)
    extent = om.Region(extent=bbox, crs=EPSG)
    dem = om.DEM(fdem, crs=4326)

    min_edge_length = 0.0025  # minimum mesh size in domain in projection
    max_edge_length = 0.10  # maximum mesh size in domain in projection
    shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
    sdf = om.signed_distance_function(shoreline)

    edge_length1 = om.feature_sizing_function(
        shoreline,
        sdf,
        max_edge_length=max_edge_length,
        crs=EPSG,
    )
    edge_length2 = om.bathymetric_gradient_sizing_function(
        dem,
        slope_parameter=5.0,
        filter_quotient=50,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        crs=EPSG,
    )
    edge_length3 = om.compute_minimum([edge_length1, edge_length2])
    edge_length3 = om.enforce_mesh_gradation(edge_length3, gradation=0.15)

    for name_, edge_length in zip(
        [
            "Feature Sizing",
            "Bathymetric Gradient",
            "Feature Sizing & Bathymetric Gradient",
        ],
        [edge_length1, edge_length3],
    ):
        print(f"Generating mesh associated with {name_}")
        edge_length_ = om.enforce_mesh_gradation(edge_length, gradation=0.15)
        generate_mesh(name_, sdf, edge_length_)

test_bathymetric_gradient_function()