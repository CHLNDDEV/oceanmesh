import os

import oceanmesh as om

# Note: global_stereo.shp has been generated using global_tag in pyposeidon
# https://github.com/ec-jrc/pyPoseidon/blob/9cfd3bbf5598c810004def83b1f43dc5149addd0/pyposeidon/boundary.py#L452
fname = os.path.join(os.path.dirname(__file__), "global", "global_latlon.shp")
fname2 = os.path.join(os.path.dirname(__file__), "global", "global_stereo.shp")


def test_global_stereo():
    # it is necessary to define all the coastlines at once:
    # the Shoreline class will the detect the biggest coastline (antartica and define it
    # as the outside boundary)

    EPSG = 4326  # EPSG:4326 or WGS84
    bbox = (-180.00, 180.00, -89.00, 90.00)
    extent = om.Region(extent=bbox, crs=4326)

    min_edge_length = 0.5  # minimum mesh size in domain in meters
    max_edge_length = 2  # maximum mesh size in domain in meters
    shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
    sdf = om.signed_distance_function(shoreline)
    edge_length0 = om.distance_sizing_function(shoreline, rate=0.11)
    edge_length1 = om.feature_sizing_function(
        shoreline,
        sdf,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        crs=EPSG,
    )

    edge_length = om.compute_minimum([edge_length0, edge_length1])
    edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.09, stereo=True)

    # once the size functions have been defined, wed need to mesh inside domain in
    # stereographic projections. This is way we use another coastline which is
    # already translated in a sterographic projection
    shoreline_stereo = om.Shoreline(fname2, extent.bbox, min_edge_length, stereo=True)
    domain = om.signed_distance_function(shoreline_stereo)

    points, cells = om.generate_mesh(domain, edge_length, stereo=True, max_iter=100)

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # apply a Laplacian smoother
    points, cells = om.laplacian2(points, cells, max_iter=100)

    # plot
    fig, ax, _ = edge_length.plot(
        holding=True,
        plot_colorbar=True,
        stereo=True,
        vmax=max_edge_length,
    )

    ax.triplot(points[:, 0], points[:, 1], cells, color="gray", linewidth=0.5)
    shoreline_stereo.plot(ax=ax)
