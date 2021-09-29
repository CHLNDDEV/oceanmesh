import os

import matplotlib.pyplot as pyplot
import oceanmesh

shp = os.path.join(os.path.dirname(__file__), "islands.shp")


def test_rect():
    bbox = (0.4, 1.6, -0.6, 0.6)

    min_edge_length = 0.02  # 2.0e3  # h0
    max_edge_length = 0.1  # 10.0e3

    region = oceanmesh.Region(bbox, 4326)
    shore = oceanmesh.Shoreline(shp, region.bbox, min_edge_length)
    edge_length = oceanmesh.distance_sizing_function(
        shore, rate=0.10, max_edge_length=max_edge_length
    )
    domain = oceanmesh.signed_distance_function(shore)

    points, cells = oceanmesh.generate_mesh(domain, edge_length)

    pyplot.figure(1)
    pyplot.clf()
    pyplot.triplot(points[:, 0], points[:, 1], cells, "-", lw=0.5, color="0.5")
    pyplot.plot(shore.boubox[:, 0], shore.boubox[:, 1], "-", color="r", markersize=0)
    pyplot.plot(shore.inner[:, 0], shore.inner[:, 1], ".", color="gray", markersize=2)
    pyplot.plot(
        shore.mainland[:, 0], shore.mainland[:, 1], "-", color="green", linewidth=0.5
    )
    pyplot.gca().axis("equal")
    pyplot.show()
