import os

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import oceanmesh as om

fname = os.path.join(os.path.dirname(__file__), "GSHHS_i_L1.shp")
EPSG = 4326  # EPSG:4326 or WGS84


def test_multiscale_overlap():
    extent1 = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
    min_edge_length1 = 1000.0e-5  # ~1.0km
    max_edge_length1 = 12500.0e-5  # ~12.5km

    bbox2 = np.array(
        [
            [-73.9481, 40.6028],
            [-74.0186, 40.5688],
            [-73.9366, 40.5362],
            [-73.7269, 40.5626],
            [-73.7231, 40.6459],
            [-73.8242, 40.6758],
            [-73.9481, 40.6028],
        ],
        dtype=float,
    )
    extent2 = om.Region(extent=bbox2, crs=EPSG)
    min_edge_length2 = 600.0e-5  # ~500m

    bbox3 = np.array(
        [
            [-73.8262, 40.6500],
            [-73.8230, 40.6000],
            [-73.7500, 40.6030],
            [-73.7450, 40.6430],
            [-73.8262, 40.6500],
        ],
        dtype=float,
    )
    extent3 = om.Region(extent=bbox3, crs=EPSG)
    min_edge_length3 = 300.0e-5  # ~300m

    s1 = om.Shoreline(fname, extent1.bbox, min_edge_length1)
    sdf1 = om.signed_distance_function(s1)
    el1 = om.distance_sizing_function(s1, max_edge_length=max_edge_length1)

    s2 = om.Shoreline(fname, extent2.bbox, min_edge_length2)
    sdf2 = om.signed_distance_function(s2)
    el2 = om.distance_sizing_function(s2)

    s3 = om.Shoreline(fname, extent3.bbox, min_edge_length3)
    sdf3 = om.signed_distance_function(s3)
    el3 = om.distance_sizing_function(s3)

    # Control the element size transition
    # from coarse to fine with the kwargs prefixed with `blend`
    points, cells = om.generate_multiscale_mesh(
        [sdf1, sdf2, sdf3], [el1, el2, el3], blend_width=1000, blend_max_iter=100
    )
    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    # remove singly connected elements (elements
    # connected to only one other element)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)
    # remove poor boundary elements with quality < 15%
    points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
    # apply a Laplacian smoother that preservers the mesh size distribution
    points, cells = om.laplacian2(points, cells)

    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.1)
    plt.figure(figsize=[4.8, 6.4])

    ax = plt.subplot(gs[0, 0])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")
    ax.plot(bbox3[:, 0], bbox3[:, 1], "r--")

    ax = plt.subplot(gs[1, 0])
    buf = 0.07
    ax.set_xlim([min(bbox2[:, 0]) - buf, max(bbox2[:, 0]) + buf])
    ax.set_ylim([min(bbox2[:, 1]) - buf, max(bbox2[:, 1]) + buf])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")

    ax = plt.subplot(gs[2, 0])
    buf = 0.07
    ax.set_xlim([min(bbox3[:, 0]) - buf, max(bbox3[:, 0]) + buf])
    ax.set_ylim([min(bbox3[:, 1]) - buf, max(bbox3[:, 1]) + buf])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox3[:, 0], bbox3[:, 1], "r--")

    plt.show()


def test_multiscale_non_overlap():
    extent1 = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
    min_edge_length1 = 1000.0e-5  # ~1.0km
    max_edge_length1 = 12500.0e-5  # ~12.5km

    bbox2 = np.array(
        [
            [-73.9481, 40.6028],
            [-74.0186, 40.5688],
            [-73.9366, 40.5362],
            [-73.7269, 40.5626],
            [-73.7231, 40.6459],
            [-73.8242, 40.6758],
            [-73.9481, 40.6028],
        ],
        dtype=float,
    )
    extent2 = om.Region(extent=bbox2, crs=EPSG)
    min_edge_length2 = 500.0e-5  # ~500m

    bbox3 = np.array(
        [
            [-71.4700, 41.8500],
            [-71.4700, 41.4000],
            [-71.1500, 41.4000],
            [-71.1500, 41.8000],
            [-71.4700, 41.8500],
        ],
        dtype=float,
    )
    extent3 = om.Region(extent=bbox3, crs=EPSG)
    min_edge_length3 = 500.0e-5  # ~500m

    s1 = om.Shoreline(fname, extent1.bbox, min_edge_length1)
    sdf1 = om.signed_distance_function(s1)
    el1 = om.distance_sizing_function(s1, max_edge_length=max_edge_length1)

    s2 = om.Shoreline(fname, extent2.bbox, min_edge_length2)
    sdf2 = om.signed_distance_function(s2)
    el2 = om.distance_sizing_function(s2)

    s3 = om.Shoreline(fname, extent3.bbox, min_edge_length3)
    sdf3 = om.signed_distance_function(s3)
    el3 = om.distance_sizing_function(s3)

    # Control the element size transition from
    # coarse to fine with the kwargs prefixed with `blend`.
    # Function objects must appear in order of descending `min_edge_length`.
    points, cells = om.generate_multiscale_mesh(
        [sdf1, sdf2, sdf3], [el1, el2, el3], blend_width=1000, blend_max_iter=100
    )
    # remove degenerate mesh faces and other common problems in the mesh
    points, cells = om.make_mesh_boundaries_traversable(points, cells)
    # remove singly connected elements (elements
    # connected to only one other element)
    points, cells = om.delete_faces_connected_to_one_face(points, cells)
    # remove poor boundary elements with quality < 15%
    points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
    # apply a Laplacian smoother that preservers the mesh size distribution
    points, cells = om.laplacian2(points, cells)

    triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
    gs = gridspec.GridSpec(3, 1)
    gs.update(wspace=0.1)
    plt.figure(figsize=[4.8, 6.4])

    ax = plt.subplot(gs[0, 0])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")
    ax.plot(bbox3[:, 0], bbox3[:, 1], "r--")

    ax = plt.subplot(gs[1, 0])
    buf = 0.07
    ax.set_xlim([min(bbox2[:, 0]) - buf, max(bbox2[:, 0]) + buf])
    ax.set_ylim([min(bbox2[:, 1]) - buf, max(bbox2[:, 1]) + buf])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")

    ax = plt.subplot(gs[2, 0])
    buf = 0.07
    ax.set_xlim([min(bbox3[:, 0]) - buf, max(bbox3[:, 0]) + buf])
    ax.set_ylim([min(bbox3[:, 1]) - buf, max(bbox3[:, 1]) + buf])
    ax.set_aspect("equal")
    ax.triplot(triang, "-", lw=0.5)
    ax.plot(bbox3[:, 0], bbox3[:, 1], "r--")

    plt.show()
