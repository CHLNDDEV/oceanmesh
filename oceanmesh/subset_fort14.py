import os
import time
import datetime
import numpy as np
import pandas as pd
import rtree
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import islice
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

def readNodes_fort14(f14):
    ''' Fx to read the fort.14 nodes as a pandas dataframe
        Parameters
            f14: string
               full path of the fort.14 file
        Returns
            Nodes: pandas dataframe
    '''
    with open(f14) as fin:
        head = list(islice(fin, 2))
        data = [int(x) for x in head[1].split()]
    nodes = pd.read_csv(f14, skiprows = 2, nrows = data[1], names = ['x', 'y', 'z'], delim_whitespace = True)
    nodes.index = [x - 1 for x in nodes.index]
    return nodes
    
def pointsInsidePoly(points, polygon):
    ''' Get the subset of a cloud of points which are inside a polygon
        Used in filledContours2gpd.
        Parameters
            points: list
                list of pair of coordinates as tuples
            polygon: list
                list of coordinates of the polygon vertices as tuples
        Returns
            cont: array
                array with booleans
    '''
    p = mpl.path.Path(polygon)
    cont = p.contains_points(points)
    return cont
    
def fort14togdf(filein, epsgIn, epsgOut):
    ''' Write adcirc mesh from fort.14 file as GeoDataFrame and extract centroid of each element. 
        Used in the downscaling process
        Parameters:
            filein: str
                full path of the fort.14 file
            epsgIn: int
                coordinate system of the adcirc input
            epsgOut: int
                coordinate system of the output shapefile
        Returns
            gdf: GeoDataFrame
                GeoDataFrame with polygons as geometry and more info such as: area, representative
                element size, centroids coordinates, and vertices
    '''
    ## read only the two first lines of the file to get the number of elements and nodes
    with open(filein) as fin:
        head = list(islice(fin, 2))
        data = [int(x) for x in head[1].split()]
    ## read nodes
    nodes = np.loadtxt(filein, skiprows = 2, max_rows = data[1], usecols = (1, 2, 3))
    ## read elements
    elem = np.loadtxt(filein, skiprows = 2 + data[1], max_rows = data[0], usecols = (2, 3, 4)) - 1
    x = nodes[:, 0]
    y = nodes[:, 1]
    z = nodes[:, 2]
    ## matplotlib triangulation
    tri = mpl.tri.Triangulation(x, y, elem)
    ## select the coordinate of each vertex
    xvertices = x[tri.triangles[:]]
    yvertices = y[tri.triangles[:]]
    zvertices = z[tri.triangles[:]]
    listElem = np.stack((xvertices, yvertices), axis = 2)
    ## define polygons and GeoDataFrame
    pols = [Polygon(x) for x in listElem]
    gdf = gpd.GeoDataFrame(geometry = pols, crs = 4326)
    
    ## change crs
    if epsgIn == epsgOut:
        pass
    else:
        gdf = gdf.to_crs(epsgOut)
    
    ## get centroids and vertices coordinatess
    gdf['zmean'] = -1*zvertices.mean(axis = 1)
    gdf['centX'] = xvertices.mean(axis = 1)
    gdf['centY'] = yvertices.mean(axis = 1)
    gdf['v1'] = elem[:, 0]
    gdf['v2'] = elem[:, 1]
    gdf['v3'] = elem[:, 2]
    gdf['id'] = range(len(gdf))
    
    ## compute area and presentative length if the output crs is not lat/lon
    if epsgOut == 4326:
        pass
    else:
        gdf['repLen'] = [np.round(geom.length/3, 3) for geom in gdf.geometry]
        gdf['minLen'] = [np.min([distance.euclidean(pi, pj) for pi, pj in zip(geom.boundary.coords[:-1], geom.boundary.coords[1:])]) for geom in gdf.geometry]
        gdf['elemArea'] = [np.round(geom.area, 3) for geom in gdf.geometry]
    
    return gdf
    
def readSubDomain(subDomain, epsg):
    ''' Read a user specified subdomain and transform it to GeoDataFrame
        Parameters
            subDomain: str or list
                complete path of the subdomain polygon kml or shapelfile, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates
            epsg: int
                epsg code of the used system of coordinates
        Returns
            gdfSubDomain: GeoDataFrame
                user specified subdomain as geodataframe
    '''
    if type(subDomain) == str:
        ## kml or shp file
        if subDomain.endswith('.shp') or subDomain.endswith('.gpkg'):
            gdfSubDomain = gpd.read_file(subDomain)
            ## get exterior coordinates of the polygon
            xAux, yAux = gdfSubDomain.geometry[0].exterior.coords.xy
            extCoords = list(zip(xAux, yAux))
            poly = Polygon(extCoords)
            gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
        
        elif subDomain.endswith('.kml'):
            gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
            gdfSubDomain = gpd.read_file(subDomain, driver = 'KML')
            ## get exterior coordinates of the polygon
            xAux, yAux = gdfSubDomain.geometry[0].exterior.coords.xy
            extCoords = list(zip(xAux, yAux))
            poly = Polygon(extCoords)
            gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
        
        else: ## try read DEMs
            try:
                r = rxr.open_rasterio(subDomain)
                bbox = r.rio.bounds()
                ulLon, ulLat, lrLon, lrLat = bbox[0], bbox[3], bbox[2], bbox[1]
                extCoords = [(ulLon, ulLat), (ulLon, lrLat), (lrLon, lrLat), (lrLon, ulLat), (ulLon, ulLat)]
                poly = Polygon(extCoords)
                gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)

            except:
                print('Only shape, geopackage, kml formats and rasters are suported for sub domain generation!')
                sys.exit(-1)            
    
    elif type(subDomain) == list and len(subDomain) == 4:
        ## only UL lon, UL lat, LR lon LR lat
        ulLon, ulLat, lrLon, lrLat = subDomain
        ## define list with exterior coordinates
        extCoords = [(ulLon, ulLat), (ulLon, lrLat), (lrLon, lrLat), (lrLon, ulLat), (ulLon, ulLat)]
        ## define shapely polygon
        poly = Polygon(extCoords)
        ## define gdf
        gdfSubDomain = gpd.GeoDataFrame(geometry = [poly], crs = epsg)
    else:
        print('subDomain must be the path of a kml or shapefile, or a list with the coordinates of ' \
              'the upper left and lower right corners of a box')
        sys.exit(-1)
    
    return gdfSubDomain
    
def subsetMeshGdf(gdf, nodes, mask):
    '''  Subsets a geodataframe based on a mask polygon and returns matplotlib triangulation,
         the nodes outside the mask, and a geodataframe with the mesh subset. It is important to note
         that the part of the mesh inside the mask is removed.
         
         Parameters
            gdf: geopandas GeoDataFrame
                mesh gdf, each element is an individual polygon. Output of fort14togdf function
            nodes: pandas dataframe
                mesh nodes, output of readNodes_fort14 function
            mask: geopandas GeoDataFrame
                mask gdf, output of readSubDomain function.
        Returns
            newMeshTri: matplotlib triangulation object
                subset of the mesh triangulation
            meshSub: geopandas GeoDataFrame
                mesh subset gdf, each element is an individual polygon.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
    '''
    # Extract exterior coordinates of the mask polygon
    xAux, yAux = mask.geometry[0].exterior.coords.xy
    extCoords = list(zip(xAux, yAux))
    # Get centroids of mesh elements
    centroids = list(zip(gdf['centX'], gdf['centY']))
    # Determine which centroids are inside the mask polygon
    inside = pointsInsidePoly(centroids, extCoords)
    # Get indices of elements to keep (outside the mask polygon)
    # starts from 0 and not from 1 as ADCIRC nodes
    centOutside = np.where(inside == False)[0]
    # Subset the geodataframe based on the selected elements, the index starts from 0
    # so iloc or loc gives same result
    meshSub = gdf.iloc[centOutside, :]
    # get nodes of the elements to keep, starts from 0
    nodesOutside = meshSub.loc[:, ['v1', 'v2', 'v3']].values.reshape(-1)
    # remove repeated nodes
    nodesOutside = np.unique(nodesOutside)
    # get coordinates of nodes to keep, the index of the series correspond to the
    # full mesh
    xNodesOutside = nodes.iloc[nodesOutside, 0]
    yNodesOutside = nodes.iloc[nodesOutside, 1]
    zNodesOutside = nodes.iloc[nodesOutside, 2]
    # Create a lookup table for renumering the mesh
    aux = {value: index for index, value in enumerate(list(nodesOutside))}
    # find index of each element vertex in the list nodes to keep
    newV = [(aux[x], aux[y], aux[z]) for x, y, z in zip(meshSub['v1'], meshSub['v2'], meshSub['v3'])]
    # Create a triangulation based on the subsetted nodes and elements
    newMeshTri = mpl.tri.Triangulation(xNodesOutside, yNodesOutside, newV)
    # add new element's nodes to the geodataframe
    meshSub[['v1u', 'v2u', 'v3u']] = newV
    meshSub.index = range(len(meshSub))
    ## dataframe with new mesh nodes
    dfNodesOutside = pd.DataFrame({'x': xNodesOutside.values,
                                    'y': yNodesOutside.values,
                                    'z': zNodesOutside.values,
                                    'orgIndex': xNodesOutside.index})

    return newMeshTri, meshSub, dfNodesOutside

def readBCfort14(f14, nodes, epsg = 4326):
    ''' Reads boundary condition information from a fort.14 file and returns a GeoDataFrame and a dictionary
        Parameters
            f14: string
                full path of the fort.14 file
            nodes: pandas dataframe
                mesh nodes, output of readNodes_fort14 function
            epsg: int. Default 4326 (lat/lon)
                coordinate reference system
        Returns
            gdfBC: geopandas GeoDataFrame
                geodataframe with fort.14 boundary conditions
            dctBC: dictionary
                boundary conditions nodes ID
    '''
    with open(f14) as fin:
        ## get header of the fort.14: number of elements and nodes
        head = list(islice(fin, 2))
        data = [int(x) for x in head[1].split()]
        ## read lines with BC information
        lines = fin.readlines()[data[0]+data[1]:]
        ## dictionary to store the data
        dctBC = {'n_open_bound': int(lines[0].split()[0]), 'total_nodes_open_bc': int(lines[1].split()[0])}
        lines = lines[2:]
        ob = 0 ## open boundary counter
        lb = 0 ## line boundary counter
        
        while len(lines) > 0:
            ## ob is for open boundary
            if ob < dctBC['n_open_bound']:
               # Read open boundary information
                nn = int(lines[0].split()[0]) # Number of nodes in the boundary condition
                bc = [int(x)-1 for x in lines[1:nn+1]] # Node index of the boundary condition, starts from 0 as python indices
                dctBC[f'bc_open_bound_{ob}'] = bc # Store the boundary condition in the dictionary
                aux = 0 # flag that helps when reading land boundaries
                lines = lines[len(bc)+1:] # remove the lines with the BC info already stored
                ob += 1 # update counter
            else:
                if aux == 0:
                    # first time reading land BC
                    dctBC['n_land_bound'] = int(lines[0].split()[0]) # number of land boundaries
                    dctBC['total_nodes_land_bc'] = int(lines[1].split()[0]) # land boundaries total nodes
                    aux += 1 # update flag
                    lines = lines[2:] # remove analyzed lines
                else:
                    nn = int(lines[0].split()[0]) # number of nodes of the current land boundary
                    bc = [int(x.split()[0]) - 1 for x in lines[1:1+nn]] # Node index of the boundary condition, starts from 0 as python indices
                    dctBC[f'bc_land_bound_{lb}'] = bc # store the data
                    lb += 1 # update counter
                    lines = lines[nn+1:] # remove analyzed lines
    
    ## create geodataframe
    nBC, lBC, tBC = [], [], []
    for key in [x for x in dctBC.keys() if x.startswith('bc_')]:
        ## check if BC is closed or open
        bc = dctBC[key]
        if bc[0] == bc[-1]: ## bc is closed
            ## define shapely Polygon
            geom = Polygon(list(zip(nodes.loc[dctBC[key], 'x'],
                                    nodes.loc[dctBC[key], 'y'])))
            dummy = 1 ## flag for closed BC
        else: ## bc is open --> ocean or main land boundary
            geom = LineString(list(zip(nodes.loc[dctBC[key], 'x'],
                                    nodes.loc[dctBC[key], 'y'])))
            dummy = 0
        nBC.append(key) ## name
        lBC.append(geom) ## geometries
        tBC.append(dummy) ## type

    gdfBC = gpd.GeoDataFrame(geometry = lBC, crs = epsg)
    gdfBC['bc_name'] = nBC
    gdfBC['bc_closed'] = tBC

    return gdfBC, dctBC

def renumClosedBCs(gdf, mask, dfNodesNew):
    ''' Update the numbering of the closed land boundary conditions
        Parameters
            gdf: geopandas GeoDataFrame
                boundary conditions gdf, output of readBCfort14
            mask: geopandas GeoDataFrame
                mask gdf, output of readSubDomain function.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
        Returns
            dctBC_closed: dictionary
                updated closed boundary conditions nodes number
            gdfBC_closed: geopandas GeoDataFrame
                updated closed boundary conditions gdf
    '''
    ## iterate through closed BC to see if they are inside or outside the new boundary
    gdfBC_closed = gdf[gdf['bc_closed'] == 1]
    auxList = []
    for bc in gdfBC_closed['geometry']:
        ## subDom represents the part of the mesh I want
        ## to exclude from the fort.14
        within = bc.within(mask['geometry'][0])
        if within == True:
            auxList.append(False)
        else:
            auxList.append(True)

    gdfBC_closed['inNewDom'] = auxList

    ## get the node ID of the BCs, the node's ID are related to the new mesh
    ## create lookup table, starts from 0
    coordsNodesOutside = list(zip(dfNodesNew['x'], dfNodesNew['y']))
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(coordsNodesOutside)}
    dctBC_closed = {}

    for i in gdfBC_closed[gdfBC_closed['inNewDom'] == True].index:
        coords = list(gdfBC_closed.loc[i, 'geometry'].exterior.coords)
        # Find the indices of the target tuples
        indices = [lookup_table.get(tuple_val) for tuple_val in coords]
        dctBC_closed[gdfBC_closed.loc[i, 'bc_name']] = indices

    return dctBC_closed, gdfBC_closed[gdfBC_closed['inNewDom'] == True]

def renumOceanBC(gdf, dfNodesNew, sortBy=1, rev=False):
    ''' Update the numbering of the ocean boundary condition. The ocean BC nodes are sorted depending
        on the mesh orientation. E.g. if the mesh is aligned with N-S and the BC is eastwards of the coast
        (like NA meshes), the nodes are sorted by latitude in ascending order.
        Parameters
            gdf: geopandas GeoDataFrame
                boundary conditions gdf, output of readBCfort14
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            sortBy: int. Default 1
                If 1, nodes are sorted by latitude since BC is aligned with N-S (vertical).
                If 0, nodes are sorted by longitude since BC is aligned with W-E (horizontal).
            rev: boolean. Default False
                If False, nodes are sorted in increasing order. This is neede 
        Returns
            dfOpen: pandas dataframe
                renumbered ocean boundary condition
            gdfBC_open: geopandas GeoDataFrame
                updated closed boundary conditions gdf

    '''
    ## for now the code will assume the ocean boundary is not modified and the mask does not overlay with it
    ## get only the open bcs
    gdfBC_open = gdf[gdf['bc_closed'] == 0]

    ## get ocean bc
    oceanOpen = gdfBC_open[gdfBC_open['bc_name'] == 'bc_open_bound_0']['geometry'][0]
    ## get list of coordinate tuples
    oceanOpenCoords = list(oceanOpen.coords)
    ## sort the nodes depending in mesh orientation
    oceanOpenCoords = sorted(oceanOpenCoords, key = lambda x: x[sortBy], reverse = rev)
    
    ## get the node ID of the BCs, the node's ID are related to the new mesh
    ## create lookup table
    coordsNodesOutside = list(zip(dfNodesNew['x'], dfNodesNew['y']))
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(coordsNodesOutside)}
    ## find id of each bc node. ADCIRC numbering starts from 1
    indices = [lookup_table.get(tuple_val) for tuple_val in oceanOpenCoords]
    dfOpen = dfNodesNew.iloc[indices, :]
    
    return dfOpen, gdfBC_open.iloc[[0], :]

def renumMainlandBC(gdfNew, gdfOcean, dfOcean, dfNodesNew, epsg = 4326):
    ''' Update the numbering of the mainland boundary condition
        Parameters
            gdfNew: geopandas GeoDataFrame
                mesh subset gdf, each element is an individual polygon. Output of subsetMeshGdf function
            gdfOcean:geopandas GeoDataFrame
                updated closed boundary conditions gdf. Output of updateOceanBC
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            epsg: int. Default 4326
                coordinate reference system
        Returns
            dfMainlandBC: pandas DataFrame
                BC with the updated nodes id
    '''
    ## get outer polygon of the mesh geodataframe
    outerPolNewMesh = gdfNew['geometry'].unary_union
    gdfOuterPolNewMesh = gpd.GeoDataFrame(geometry = [outerPolNewMesh], crs = epsg).boundary

    ## here I assumed that the linestring with more nodes is the outer boundary (ocean + mainland)
    max_nodes = 0
    max_line = None
    for line in gdfOuterPolNewMesh[0].geoms:
        num_nodes = len(line.coords)
        # Check if the current LineString has more nodes than the previous maximum
        if num_nodes > max_nodes:
            max_nodes = num_nodes
            max_line = line

    ## geodataframe with mainland + ocean bc
    mainBoundary = gpd.GeoDataFrame(geometry = [max_line], crs = epsg)
    ## ocean boundary condition linestring
    lsOcean = gdfOcean.loc[0, 'geometry']
    ## outer polygon boundary as linestring
    lsBound = mainBoundary.loc[0, 'geometry']
    ## get coordinates, list with tuples [(lon0, lat0), (lon1, lat1), ...]
    lsOceanCoords = list(zip(dfOcean['x'], dfOcean['y']))
    lsBoundCoords = list(lsBound.coords)
    ## define list of linestrings for the mesh ordering
    linesAll = [LineString((x, y)) for x, y in zip(lsBoundCoords[:-1], list(lsBoundCoords)[1:])]
    linesOcean = [LineString((x, y)) for x, y in zip(list(lsOceanCoords)[:-1], list(lsOceanCoords)[1:])]

    ## geodataframe with ocean bc
    gdfOcean = gpd.GeoDataFrame(geometry = linesOcean, crs = epsg)
    ## geodataframe with full boundary
    gdfAll = gpd.GeoDataFrame(geometry = linesAll, crs = epsg)
    ## geodataframe with mainland (difference between full boundary and ocean bc)
    gdfLand = gpd.overlay(gdfAll, gdfOcean, how = 'difference')

    ## start the mainland with the last point of the ocean bc to satisfy the counter clockwise ordering
    mainlandCounter = [Point(lsOceanCoords[-1])]

    # Build a spatial index for the LineString geometries
    spatial_index = rtree.index.Index()
    for i, geometry in enumerate(gdfLand['geometry']):
        spatial_index.insert(i, geometry.bounds)

    i = 0
    while len(mainlandCounter) < len(gdfLand)+1:
        # Find the nearest LineString to the last point of the mainlandCounter using the spatial index
        nearest_idx = list(spatial_index.nearest(mainlandCounter[-1].bounds, 1))[0]
        nearestLine = gdfLand.iloc[nearest_idx]['geometry']
        
        # Add the first point of the nearest LineString to mainlandCounter
        mainlandCounter.append(Point(nearestLine.coords[0]))
        
        # Remove the analyzed LineString from the spatial index
        spatial_index.delete(nearest_idx, nearestLine.bounds)
        i+=1

    ## mainlandCounter has the bound of each lineString, but we only need the starting point of each linestring
    mainlandCounterCoords = [x.bounds[:2] for x in mainlandCounter]
    # Create a dictionary lookup table for the indices
    lookup_table = {tuple_val: index for index, tuple_val in enumerate(zip(dfNodesNew['x'], dfNodesNew['y']))}
    # Find the indices of the target tuples
    indicesMainland = [lookup_table.get(tuple_val) for tuple_val in mainlandCounterCoords]
    dfMainlandBC = dfNodesNew.iloc[indicesMainland, :]
    
    return dfMainlandBC

def writeFort14(f14in, f14out, gdfNew, dfNodesNew, dfOpen, dctClosed, mainlandBC):
    ''' Write the fort.14
        Parameters
            f14in: string
                full path of the original fort.14. It is used only to get the header
            f14out: string
                full path of the output fort.14
            gdfNew: geopandas GeoDataframe
                mesh subset gdf, each element is an individual polygon.
            dfNodesOutside: pandas dataframe
                coordinates of the nodes outside the mask (new mesh nodes). Dataframe index
                starts from 0, has x, y and z values, and the index of the node in the original mesh
            dfOpen: pandas dataframe
                renumbered ocean boundary condition
            dctClosed: dictionary
                updated closed boundary conditions nodes number
            mainlandBC: pandas DataFrame
                BC nodes with the updated ID
        Returns
            None
    '''

    ## get original fort.14 header
    with open(f14in, 'r') as fin:
        header = list(islice(fin, 1))[0][:-1]
    
    now = datetime.datetime.now()
    nowStr = now.strftime("%Y/%m/%d %H:%M:%S")
    
    ## start writing new fort.14
    with open(f14out, 'w') as fout:
        ## write new header
        fout.write(f'{header} modified with fort14Subset on {nowStr}\n')
        ## write number of elements and nodes
        fout.write(f'{len(gdfNew)} {len(dfNodesNew)}\n')
        ## write nodes
        for i in dfNodesNew.index:#i, (xi, yi, zi) in enumerate(zip(xNodes, yNodes, zNodes)):
            xi = dfNodesNew.loc[i, 'x']
            yi = dfNodesNew.loc[i, 'y']
            zi = dfNodesNew.loc[i, 'z']
            fout.write(f"   {i+1:7}  {xi:13.10f}  {yi:13.10f}  {zi:14.10f}\n")
        ## write triangles
        for i in gdfNew.index:
            v1 = gdfNew.loc[i, 'v1u'] + 1
            v2 = gdfNew.loc[i, 'v2u'] + 1
            v3 = gdfNew.loc[i, 'v3u'] + 1
            fout.write(f"{i+1:7} 3 {v1} {v2} {v3}\n")
    
        ## start BC section
        # write number of open boundaries and total nodes
        fout.write("1 = Number of open boundaries\n")
        # get total number of open boundary nodes
        # total_nodes_open_bc = sum(len(lst) for lst in dctOpen.values())
        fout.write(f"{len(dfOpen)} = Total number of open boundary nodes\n")
    
        # write ocean boundary condition
        fout.write(f"{len(dfOpen)} 20 = Number of nodes for open boundary 1\n")
        for n in dfOpen.index:
            fout.write(f'{n + 1}\n')
        
        # write number of land boundaries and total nodes
        fout.write(f"{1 + len(dctClosed.keys())} = Number of land boundaries\n")
        # get total number of land boundary nodes
        total_nodes_land_bc = sum(len(lst) for lst in dctClosed.values()) + len(mainlandBC)
        fout.write(f"{total_nodes_land_bc} = Total number of land boundary nodes\n")
        
        # write main land boundary
        fout.write(f"{len(mainlandBC)} 20 = Number of nodes for land boundary 1\n")
        for n in mainlandBC.index:
            fout.write(f'{n + 1}\n')
            # fout.write(f"{int(mainlandBC.loc[n, 'index'])+1}\n")
        
        ## closed land boundaries
        for ik, k in enumerate(dctClosed.keys()):
            fout.write(f"{len(dctClosed[k])} 21 = Number of nodes for land boundary {ik+2}\n")
            for n in dctClosed[k]:
                fout.write(f'{n + 1}\n')

def fort14Subset(f14in, subDomain, f14out, epsg=4326, sortBy=1, rev=False):
    ''' Create a subset of a fort.14 using a shapefile as mask to remove the elements. 
        The code has some limitations since it has been tested only with meshes of the entire North Atlantic.
            - It is assumed that the ocean BC goes first in the fort.14
            - Only one ocean BC
            - Mask should not intersects islands or closed boundary conditions (not tested).
            - It is assumed there are closed boundaries or islands in the domain.

        NNfort13 function can be used to create a fort.13 for the new fort.14. It uses nearest neighbor
        to interpolate the nodal attributes from the original mesh to the new one.

        Note that as the nodes are renumebered, the fort.15 tide constituents might be changed.

        Parameters
            f14in: string
                full path of the original fort.14
            subDomain: str or list
                complete path of the subdomain polygon kml, shapefile or geopackage, or list with the
                uper-left x, upper-left y, lower-right x and lower-right y coordinates
            fout: string
                full path of the output fort.14
            epsg: int. Default 4326 (lat/lon)
                coordinate reference system of the mesh and mask layer
            sortBy: int. Default 1
                If 1, nodes are sorted by latitude since BC is aligned with N-S (vertical).
                If 0, nodes are sorted by longitude since BC is aligned with W-E (horizontal).
            rev: boolean. Default False
                This is needed to sort counter clockwise the ocean boundary nodes.
                If False, nodes are sorted in increasing order. E.g. north atlantic meshes
                IF True, nodes are sorted in decreasing order. E.g. pacific ocean where the shoreline is eastwards the ocean boundary.
            
    '''
    time00 = time.time()
    ## read nodes
    print('fort.14 subset process started')
    dfNodes = readNodes_fort14(f14in)
    time01 = time.time()
    print(f'  fort.14 nodes as DataFrame: {(time01 - time00)/60:0.2f} min')
    ## convert fort.14 to gdf
    gdfMesh = fort14togdf(f14in, epsg, epsg)
    time02 = time.time()
    print(f'  fort.14 to GeoDataFrame: {(time02 - time01)/60:0.2f} min')
    ## read sub domain
    subDom = readSubDomain(subDomain, epsg)
    time03 = time.time()
    print(f'  Read subdomain: {(time03 - time02)/60:0.2f} min')
    ## subset mesh using subDomain
    _, meshSub, dfNodesNew = subsetMeshGdf(gdfMesh, dfNodes, subDom)
    time04 = time.time()
    print(f'  Subset mesh: {(time04 - time03)/60:0.2f} min')
    ## read fort.14 boundary conditions
    gdfbc, _ = readBCfort14(f14in, dfNodes)
    time05 = time.time()
    print(f'  Read fort.14 boundary conditions: {(time05 - time04)/60:0.2f} min')
    ## update the ocean boundary condition to match numbering of the subset mesh
    dfOpen, gdfOpen = renumOceanBC(gdfbc, dfNodesNew, sortBy, rev)
    time06 = time.time()
    print(f'  Update numbering ocean boundary condition: {(time06 - time05)/60:0.2f} min')
    ## update the land closed boundary conditions to match numbering of the subset mesh (islands)
    dctClosed, _ = renumClosedBCs(gdfbc, subDom, dfNodesNew)
    time07 = time.time()
    print(f'  Update numbering closed land boundary conditions: {(time07 - time06)/60:0.2f} min')
    ## update the mainland open boundary condition to match numbering of the subset mesh
    ## if more than one are merged
    dfMainland = renumMainlandBC(meshSub, gdfOpen, dfOpen, dfNodesNew, epsg)
    time07 = time.time()
    print(f'  Update numbering closed land boundary conditions: {(time07 - time06)/60:0.2f} min')
    ## write new fort.14
    writeFort14(f14in, f14out, meshSub, dfNodesNew, dfOpen, dctClosed, dfMainland)
    time08 = time.time()
    print(f'  Writing new fort.14: {(time08 - time07)/60:0.2f} min')
    print(f'Done with fort.14 subset: {(time08 - time00)/60:0.2f} min')
    
