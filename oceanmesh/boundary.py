'''
Functions to automatically label boundary segments as either 
elevation-specified (i.e., ocean) or no-flux (i.e., land) 
based on geometric and topobathymetric aspects. 

Author: Dr. Alexandra Maskell
Date: 2024-01-17
'''
import numpy as np
from sklearn.neighbors import NearestNeighbors

from oceanmesh.edges import get_winded_boundary_edges, get_boundary_edges
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


def identify_land_boundary_sections(
    points,
    cells,
    ocean_boundary,
    plot=False,
):
    """Identify the contiguous sections on the land boundary based on ocean boundary

    Parameters
    ----------
    points: numpy.ndarray
        Array of points (x,y)
    cells : numpy.ndarray
        Array of cells
    ocean_boundary : numpy.ndarray
        List of tuples of the nodes that define the ocean boundary sections
        Note these map back into the points array.
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
    boundary_sections = []
    first=True
    for idx, (s1, s2) in enumerate (ocean_boundary):
        start_idx = np.where(boundary_nodes_unmasked==s1)[0][0]
        end_idx = np.where(boundary_nodes_unmasked==s2)[0][0]
 
        if (first) & (start_idx>0):
            boundary_sections.append([0,start_idx])
            st_idx = end_idx
            first=False
        elif idx <= (len(ocean_boundary)-1):
            boundary_sections.append([st_idx,start_idx])
        
        if idx==(len(ocean_boundary)-1):
            boundary_sections.append([end_idx,len(boundary_nodes_unmasked)-1])
    

    # Plot the mesh
    if plot:
        fig, ax = plt.subplots()
        ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.1)
        
        for s1, s2 in boundary_sections:
            ax.scatter(
                points[boundary_nodes_unmasked[s1:s2], 0],
                points[boundary_nodes_unmasked[s1:s2], 1],
                5,
                c="g",
            )
            ax.set_title("Identified land boundary sections")
        plt.show()
        

    # Map back to the original node indices associated with the points array
    boundary_sections = [
        (boundary_nodes_unmasked[s1], boundary_nodes_unmasked[s2])
        for s1, s2 in boundary_sections
    ]
    
    def shift(seq, n):
        n = n % len(seq)
        return seq[n:] + seq[:n]

    ## Sort land boundaries to occur in cw order
    start_lb = np.where(boundary_sections==ocean_boundary[0][1])[0][0]
    mainland_boundary_cw = shift(boundary_sections, start_lb)

    return mainland_boundary_cw

def identify_island_boundary_sections(
    points,
    cells,
    plot=False,
    ccw=False
):
    """Identify the contiguous sections on the land boundary based on ocean boundary

    Parameters
    ----------
    points: numpy.ndarray
        Array of points (x,y)
    cells : numpy.ndarray
        Array of cells
    plot : bool, optional
        Plot the mesh and the identified boundary sections, by default False

    Returns
    --------
    boundary_nodes : list
        List of nodes that define the island boundary sections (cw direction)
        Note these map back into the points array.
    boundary_sections : list
        List of tuples of the nodes that define the island boundary sections
        Note these map back into the points array.

    """
    # Identify the nodes on the boundary of the mesh
    boundary_edges = get_winded_boundary_edges(cells)
    boundary_edges = boundary_edges.flatten()
    unique_indexes = np.unique(boundary_edges, return_index=True)[1]
    boundary_nodes_unmasked = [
        boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
    ]
    
    all_boundary_edges = get_boundary_edges(cells)
    all_boundary_edges = all_boundary_edges.flatten()
    unique_indexes = np.unique(all_boundary_edges, return_index=True)[1]
    
    all_boundary_nodes_unmasked = [
        all_boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
    ]
    
    common_elements = list(set(boundary_nodes_unmasked).intersection(set(all_boundary_nodes_unmasked)))
    all_island_boundary_nodes = boundary_nodes_unmasked + all_boundary_nodes_unmasked
    
    for item in common_elements:
        all_island_boundary_nodes = [element for element in all_island_boundary_nodes if element != item]

    island_boundary_nodes_winded = []
    choice = 0
    while True: 
        idx = all_island_boundary_nodes[choice]
        islands_boundary_edges = get_winded_boundary_edges(cells, vFirst=idx)
        islands_boundary_edges = islands_boundary_edges.flatten()
        unique_indexes = np.unique(islands_boundary_edges, return_index=True)[1]
    
        island_boundary_nodes_unmasked = [
            islands_boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
        ]
        island_boundary_nodes_unmasked.reverse()
        island_boundary_nodes_winded.append(island_boundary_nodes_unmasked)
        
        if sum(len(ele) for ele in island_boundary_nodes_winded) == len(all_island_boundary_nodes):
            break
        
        common_elements = list(set(island_boundary_nodes_unmasked).intersection(set(all_island_boundary_nodes)))
        remaining_island_nodes = island_boundary_nodes_unmasked + all_island_boundary_nodes
        for item in common_elements:
            remaining_island_nodes = [element for element in remaining_island_nodes if element != item]
        choice = np.where(all_island_boundary_nodes==remaining_island_nodes[0])[0][0]
   
    # Plot the mesh
    if plot:
        fig, ax = plt.subplots()
        ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.1)
        for i in range(len(island_boundary_nodes_winded)):
            ax.scatter(
                points[island_boundary_nodes_winded[i], 0],
                points[island_boundary_nodes_winded[i], 1],
                5,
                c="r",
            )
            ax.set_title("Identified land boundary sections")
        plt.show()
        
    # Map back to the original node indices associated with the points array
    island_boundary_sections = [
        (island_boundary_nodes_winded[i][0], island_boundary_nodes_winded[i][-1])
        for i in range(len(island_boundary_nodes_winded))
    ]
    
    #append first node to end of list:
    
    for i in range(len(island_boundary_nodes_winded)):
        island_boundary_nodes_winded[i] = [x+1 for x in island_boundary_nodes_winded[i]]
        island_boundary_nodes_winded[i].append(island_boundary_nodes_winded[i][0]) 
            
    return [island_boundary_nodes_winded, island_boundary_sections]

def split_list(l):
    index_list = [None] + [i for i in range(1, len(l)) if l[i] - l[i - 1] > 1] + [None]
    return [l[index_list[j - 1]:index_list[j]] for j in range(1, len(index_list))]

def identify_boundary_sections_knn(
    points,
    cells,
    shoreline,
    edge_length,
    plot=False,
):
    
    # Identify the nodes on the boundary of the mesh
    boundary_edges = get_winded_boundary_edges(cells)
    boundary_edges = boundary_edges.flatten()
    unique_indexes = np.unique(boundary_edges, return_index=True)[1]
    boundary_nodes_unmasked = [
        boundary_edges[unique_index] for unique_index in sorted(unique_indexes)
    ]
    
    boundary_points = points[boundary_nodes_unmasked]
    
    x = points[boundary_edges,0]
    y = points[boundary_edges,1]
    # x_mid = x.mean(axis=0)
    # y_mid = y.mean(axis=0)
    # eb_mid = np.row_stack((x_mid, y_mid)).T

    land = shoreline.mainland
    land = land[~np.isnan(land[:,0])]   
 
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(land)
    dist_lim = 25*edge_length.values.min()
    ldst,_ = knn.kneighbors(boundary_points , return_distance=True)
    ldst = ldst.min(axis=1)
    eb_class = ldst > dist_lim
    
    # count open boundaries
    ocean_boundary = []
    nope = (eb_class[:-1] < eb_class[1:]).sum()
    # find index of open boundaries
    idx_ope = split_list(np.where(eb_class==True)[0])
    for j in range(len(idx_ope)):
        if len(idx_ope[j])>3:
            idx_ope[j] = boundary_nodes_unmasked[idx_ope[j][0]:idx_ope[j][-1]]
            ocean_boundary.append((idx_ope[j][0],idx_ope[j][-1]))
    # find index of mainland boundaries
    mainland_boundary = []
    idx_mland = split_list(np.where(eb_class==False)[0])
    nmland = len(idx_mland)
    for j in range(len(idx_mland)):
        idx_mland[j] = boundary_nodes_unmasked[idx_mland[j][0]:idx_mland[j][-1]]
        mainland_boundary.append((idx_mland[j][0],idx_mland[j][-1]))
    if plot == True:
        fig, ax = plt.subplots()
        ax.plot(boundary_points[~eb_class ][:,0],boundary_points[~eb_class ][:,1],'ko')
        ax.plot(boundary_points[eb_class ][:,0],boundary_points[eb_class ][:,1],'bo') 
    
    # Define a boolean array of valid nodes
    boundary_sections = []
    first=True
    for idx, (s1, s2) in enumerate (ocean_boundary):
        start_idx = np.where(boundary_nodes_unmasked==s1)[0][0]
        end_idx = np.where(boundary_nodes_unmasked==s2)[0][0]
 
        if (first) & (start_idx>0):
            boundary_sections.append([0,start_idx])
            st_idx = end_idx
            first=False
        elif idx < (len(ocean_boundary)-2):
            boundary_sections.append([st_idx,start_idx])
        
        if idx==(len(ocean_boundary)-1):
            boundary_sections.append([end_idx,len(boundary_nodes_unmasked)-1])
    

    # Plot the mesh
    if plot:
        fig, ax = plt.subplots()
        ax.triplot(points[:, 0], points[:, 1], cells, color="k", lw=0.1)
        
        for s1, s2 in boundary_sections:
            ax.scatter(
                points[boundary_nodes_unmasked[s1:s2], 0],
                points[boundary_nodes_unmasked[s1:s2], 1],
                5,
                c="g",
            )
            ax.set_title("Identified land boundary sections")
        plt.show()
        

    # Map back to the original node indices associated with the points array
    boundary_sections = [
        (boundary_nodes_unmasked[s1], boundary_nodes_unmasked[s2])
        for s1, s2 in boundary_sections
    ]
    
    def shift(seq, n):
        n = n % len(seq)
        return seq[n:] + seq[:n]

    ## Sort land boundaries to occur in cw order
    # start_lb = np.where(boundary_sections==ocean_boundary[0][1])[0][0]
    # mainland_boundary_cw = shift(boundary_sections, start_lb)

    
    ## check for islands  - need to be ccw order (TO ADD)  
    idx_islands = []
    if len(shoreline.inner) >0:   
        all_boundary_edges = get_boundary_edges(cells)
        all_boundary_edges = all_boundary_edges.flatten()
        unique_indexes = np.unique(all_boundary_edges, return_index=True)[1]
        all_boundary_nodes_unmasked = all_boundary_edges[np.sort(unique_indexes)]
        all_boundary_points = points[all_boundary_nodes_unmasked]

        islands = shoreline.inner
        islands = np.split(islands,np.where(np.isnan(islands[:,0])==True)[0])
        islands = [y[~np.isnan(y[:,0])] for y in islands][:-1]
        
        for i in islands:
            knn = NearestNeighbors(n_neighbors=3)
            knn.fit(i)
            dist_lim = 5*edge_length.dx
            idst,_ = knn.kneighbors(all_boundary_points, return_distance=True)
            idst = idst.min(axis=1)
            ebi_class = idst < dist_lim
            idx_islands.append(np.hstack([all_boundary_nodes_unmasked[np.where(ebi_class==True)[0]],all_boundary_nodes_unmasked[np.where(ebi_class==True)[0][0]]]))
            if plot == True:
                ax.plot(all_boundary_points[ebi_class][:,0],all_boundary_points[ebi_class ][:,1],'go')
        nislands = len(idx_islands)
       
        # change index to start at 1
        for i in range(len(idx_islands)):
            idx_islands[i] = np.flip(idx_islands[i])
            idx_islands[i] = [x+1 for x in idx_islands[i]]
        
       
    return ocean_boundary,idx_islands #,ocean boundary,idx_islands


def boundary_node_list(
    cells,
    boundary_sections,
    land=False,
    ccw=True,
    node_limit = 999,
):
    """ output list of boundary nodes in c

    Parameters
    ----------
    cells : numpy.ndarray
        Array of cells
    boundary_sections : list
        List of tuples of the nodes that define the ocean boundary sections
        Note these map back into the points array.
    ccw : bool, optional
        output boundary nodes in ccw order, by default True
    node_limit : bool, optional
        output boundary nodes in list length of 1000 max, by default 999
    Returns
    --------
    boundary_nodes : list of nodes
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
    
    boundary_sections_limit=[]
    for idx, (s1, s2) in enumerate (boundary_sections):
        start_idx = np.where(boundary_nodes_unmasked==s1)[0][0]
        end_idx = np.where(boundary_nodes_unmasked==s2)[0][0]

        if (end_idx-start_idx) < node_limit:
            boundary_sections_limit.append((boundary_nodes_unmasked[start_idx],boundary_nodes_unmasked[end_idx]))
        else:
            n = int(np.ceil((end_idx-start_idx)/node_limit))
            splt_idx = start_idx.copy()
            for i in range(0,n):
                if i < (n-1):
                    boundary_sections_limit.append((boundary_nodes_unmasked[splt_idx],boundary_nodes_unmasked[splt_idx+node_limit]))
                elif i == (n-1):
                    boundary_sections_limit.append((boundary_nodes_unmasked[splt_idx],boundary_nodes_unmasked[end_idx]))
                splt_idx = splt_idx+node_limit
                
    boundary_nodes = []
    for idx, (s1, s2) in enumerate(boundary_sections_limit):
        start_idx = np.where(boundary_nodes_unmasked==s1)[0][0]
        end_idx = np.where(boundary_nodes_unmasked==s2)[0][0]
        if end_idx < (len(boundary_nodes_unmasked)-1): 
            list_nodes = boundary_nodes_unmasked[start_idx:end_idx+1]
        elif end_idx == (len(boundary_nodes_unmasked)-1):
            list_nodes = boundary_nodes_unmasked[start_idx:end_idx]
            list_nodes.append(boundary_nodes_unmasked[0])
        list_nodes = [x+1 for x in list_nodes]
        
        #if (land==True) & (idx>0):
         #   list_nodes.append(boundary_nodes[idx-1][0])
        
        if ccw==False:
            #flip node direction
            list_nodes.reverse()
       
        boundary_nodes.append(list_nodes)
               
    return boundary_nodes
