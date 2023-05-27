import numpy as np

from .edges import get_winded_boundary_edges
import matplotlib.pyplot as plt

__all__ = ["identify_ocean_boundary_sections"]


def identify_ocean_boundary_sections(
    points,
    cells,
    topobathymetry,
    depth_threshold=-50.0,
    min_nodes_threshold=10,
    plot=False,
):
    """Identify the contiguous sections on the ocean boundary based on depth
    that could be forced in a numerical model as ocean-type boundaries (e.g., elevation-specified)

    Parameters
    ----------
    points: numpy.ndarray
        Array of points (x,y)
    cells : numpy.ndarray
        Array of cells
    topobathymetry : numpy.ndarray
        Array of topobathymetry values (depth below datum negative)
    depth_threshold : float, optional
        Depth threshold to identify ocean boundary nodes, by default -50 m below the datum
    min_nodes_threshold : int, optional
        Minimum number of nodes to be considered a boundary section, by default 10
    plot : bool, optional
        Plot the mesh and the identified boundary sections, by default False

    Returns
    --------
    boundary_sections : list
        List of tuples of the nodes that define the ocean boundary sections
        Note these map back into the points array.

    """
    # Identify the nodes on the boundary of the mesh
    boundary_edges = get_winded_boundary_edges(cells)
    boundary_edges = boundary_edges.flatten()
    unique_indexes = np.unique(boundary_edges, return_index=True)[1]
    boundary_nodes_unmasked = [
        boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
    ]
    # Define a boolean array of valid nodes
    bathymetry_on_boundary = topobathymetry[boundary_nodes_unmasked]
    # Append a NaN value to the array to align with the original
    bathymetry_on_boundary = np.append(bathymetry_on_boundary, np.nan)
    stops = np.nonzero(bathymetry_on_boundary <= depth_threshold)[0]

    # Plot the mesh
    if plot:
        fig, ax = plt.subplots()
        ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.1)

    first = True
    boundary_sections = []
    start_node = None
    end_node = None
    for idx, (s1, s2) in enumerate(zip(stops[:-1], stops[1:])):
        if s2 - s1 < min_nodes_threshold:
            if first:
                start_node = s1
                first = False
            # We've reached the end of the list
            elif idx == len(stops) - 2:
                # Append the start and end nodes to the boundary sections list
                end_node = s2
                boundary_sections.append([start_node, end_node])
            # Its not the end and we haven't found a section yet
            else:
                end_node = s2
        elif s2 - s1 >= min_nodes_threshold and not first:
            # Append the start and end nodes to the boundary sections list
            boundary_sections.append([start_node, end_node])
            # Reset the start node, the last node didn't satisfy the threshold
            # and it appears we have a new section
            start_node = s1
            first = True
        # We've reached the end of the list
        elif idx == len(stops) - 2:
            # Save the end node
            end_node = s2
            # Append the start and end nodes to the boundary sections list and finish
            boundary_sections.append([start_node, end_node])
    if plot:
        for s1, s2 in boundary_sections:
            ax.scatter(
                points[boundary_nodes_unmasked[s1:s2], 0],
                points[boundary_nodes_unmasked[s1:s2], 1],
                5,
                c="r",
            )
            ax.set_title("Identified ocean boundary sections")
        plt.show()

    # Map back to the original node indices associated with the points array
    boundary_sections = [
        (boundary_nodes_unmasked[s1], boundary_nodes_unmasked[s2])
        for s1, s2 in boundary_sections
    ]
    return boundary_sections
