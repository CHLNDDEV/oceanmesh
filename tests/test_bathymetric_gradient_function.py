import oceanmesh as om


def generate_mesh(name, signed_distance, edge_length, max_iterations=500):
    points, cells = om.generate_mesh(
        signed_distance, edge_length, max_iter=max_iterations
    )
    # Makes sure the vertices of each triangle are arranged in an anti-clockwise manner
    points, cells, jx = om.fix_mesh(points, cells)
    mesh_plot(points, cells, "1 Original Mesh")

    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    mesh_plot(points, cells, "2 Degenerate Elements Removed")

    # Removes faces connected by a single face
    points, cells = om.delete_faces_connected_to_one_face(points, cells)
    mesh_plot(points, cells, "3 Deleted Faces Connected to One Face")

    # remove low quality boundary elements less than 15%
    points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
    mesh_plot(points, cells, "4 Deleted Boundary Faces")

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


if __name__ == "__main__":
    import os

    fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")
    fdem = "../datasets/EastCoast.nc"
    EPSG = 4326  # EPSG:4326 or WGS84
    bbox = (-74.4, -73.4, 40.2, 41.2)
    extent = om.Region(extent=bbox, crs=EPSG)
    dem = om.DEM(fdem, crs=4326)
    dem.plot()

    min_edge_length = 0.005  # minimum mesh size in domain in projection
    max_edge_length = 0.02  # maximum mesh size in domain in projection
    print("Creating shoreline")
    shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
    shoreline.plot(
        xlim=[extent.bbox[0], extent.bbox[1]], ylim=[extent.bbox[2], extent.bbox[3]]
    )
    print("Creating signed-distance function")
    sdf = om.signed_distance_function(shoreline)

    edge_length1 = om.feature_sizing_function(
        shoreline,
        sdf,
        max_edge_length=max_edge_length,
        crs=4326,
    )
    edge_length2 = om.bathymetric_gradient_sizing_function(
        dem,
        slope_parameter=0.01,
        filter_quotient=50,
        max_edge_length=max_edge_length,
        crs=EPSG,
    )

    edge_length3 = om.compute_minimum([edge_length2, edge_length1])

    for name_, edge_length in zip(
        [
            "Feature Sizing",
            "Bathymetric Gradient",
            "Feature Sizing & Bathymetric Gradient",
        ],
        [edge_length1, edge_length2, edge_length3],
    ):
        print(f"Generating mesh associated with {name_}")
        edge_length_ = om.enforce_mesh_gradation(edge_length, gradation=0.15)
        generate_mesh(name_, sdf, edge_length_)
