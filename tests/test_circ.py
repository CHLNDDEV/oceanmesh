import os
import matplotlib.pyplot as pyplot
import numpy
import oceanmesh
import shapefile


shp0 = os.path.join(os.path.dirname(__file__), "ocean.shp")
shp = os.path.join(os.path.dirname(__file__), "islands.shp")


def test_circ():
    with shapefile.Reader(shp0) as shpf:
        shapes = shpf.shapes()
        n = len(shapes)
        for shape in shapes:
            if n > 1:
                print(
                    "WARN, {:d} polygons in '{:s}'. Continue with item[0].".format(
                        n, os.path.basename(shp0)
                    )
                )
                break

        bbox = numpy.asarray(shape.points)
        bbox = numpy.append(bbox, [[numpy.nan, numpy.nan]], axis=0)

    del (shapes, shape, shpf)

    min_edge_length = 2.0e3  # h0
    max_edge_length = 10.0e3

    shore = oceanmesh.Shoreline(shp, bbox, min_edge_length)
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
