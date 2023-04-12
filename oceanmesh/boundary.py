import numpy as np

from .edges import get_winded_boundary_edges

__all__ = ['identify_ocean_boundary_sections']

def identify_ocean_boundary_sections(points, cells, topobathymetry, depth_threshold=50.):
    '''Identify the contiguous sections on the ocean boundary based on depth
    that could be forced in a numerical model as ocean-type boundaries (e.g., elevation-specified) 
    
    Parameters
    ----------
    points : numpy.ndarray
        Array of points
    cells : numpy.ndarray
        Array of cells
    topobathymetry : numpy.ndarray
        Array of topobathymetry values (depth below datum negative)
    depth_threshold : float, optional
        Depth threshold to identify ocean boundary nodes, by default 50 m 

    '''
    # Identify the nodes on the boundary of the mesh 
    boundary_nodes = get_winded_boundary_edges(cells)
    # Define a boolean array of valid nodes
    mask = topobathymetry[boundary_nodes] < depth_threshold # define the mask
    valid_nodes = np.where(mask)[0]

    # Define an array of valid node indices
    node_indices = np.where(mask)[0]

    # Define an empty list to store the sections
    sections = []

    # Loop through the valid node indices
    current_section = [node_indices[0]]
    for i in range(1, len(node_indices)):
        if node_indices[i] == node_indices[i-1] + 1:
            current_section.append(node_indices[i])
        else:
            sections.append((current_section[0], current_section[-1]))
            current_section = [node_indices[i]]

    # Add the final section to the list
    sections.append((current_section[0], current_section[-1]))

    return sections
