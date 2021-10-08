import matplotlib.pyplot as plt
import matplotlib.tri as tri
import oceanmesh as om


def test_grade():
    fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

    region = om.Region((-75.000, -70.001, 40.0001, 41.9000), 4326)
    min_edge_length = 0.01

    shore = om.Shoreline(fname, region.bbox, min_edge_length)

    edge_length = om.distance_sizing_function(shore, rate=0.35)

    test_edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.20)
    test_edge_length.plot(show=False, filename="test_grade_edge_length.png")

    domain = om.signed_distance_function(shore)

    points, cells = om.generate_mesh(domain, test_edge_length, max_iter=100)

    points, cells = om.make_mesh_boundaries_traversable(points, cells)

    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # plot
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(triang, "-", lw=1)
    plt.show()
