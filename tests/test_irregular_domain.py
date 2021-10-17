import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import oceanmesh as om


def test_irregular_domain():
    fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

    # New York Lower Bay and Jamaica Bay
    bbox = np.array(
        [
            [-74.1588, 40.5431],
            [-74.1215, 40.4847],
            [-74.0261, 40.4660],
            [-73.9369, 40.5034],
            [-73.8166, 40.5104],
            [-73.7524, 40.5711],
            [-73.7627, 40.6669],
            [-73.8436, 40.6809],
            [-73.9473, 40.6552],
            [-74.0883, 40.6155],
            [-74.1588, 40.5431],
        ]
    )

    min_edge_length = 0.001

    region = om.Region(bbox, 4326)
    shore = om.Shoreline(fname, region.bbox, min_edge_length)
    shore.plot(file_name="test_irregular_domain.png", show=False)

    edge_length = om.distance_sizing_function(shore, max_edge_length=0.01)

    domain = om.signed_distance_function(shore)

    points, cells = om.generate_mesh(domain, edge_length, max_iter=50)

    points, cells = om.make_mesh_boundaries_traversable(points, cells)

    points, cells = om.delete_faces_connected_to_one_face(points, cells)

    # plot
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    ax.triplot(triang, "-", lw=1)
    plt.savefig("test_irregular_domain_mesh.png")
    plt.show()
