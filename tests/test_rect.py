import os,sys,time
import matplotlib.pyplot as pyplot
import numpy
import meshio
from oceanmesh import (
    Shoreline,
    distance_sizing_function,
    signed_distance_function,
    generate_mesh,
    make_mesh_boundaries_traversable,
    delete_faces_connected_to_one_face,
    fix_mesh
)

bbox = (0.4,1.6,-0.6,0.6)
shp='islands.shp'

min_edge_length = 2.e3  # h0
max_edge_length = 10.e3

shore = Shoreline(shp, bbox, min_edge_length)
edge_length = distance_sizing_function(shore,rate=0.10, max_edge_length=max_edge_length)
domain = signed_distance_function(shore)

points, cells = generate_mesh(domain, edge_length)

pyplot.figure(1)
pyplot.clf()
pyplot.triplot(points[:,0],points[:,1],cells,'-',lw=0.5,color='0.5')
pyplot.plot(shore.boubox[:,0],shore.boubox[:,1],'x',color='r',markersize=2)
pyplot.plot(shore.inner[:,0],shore.inner[:,1],'.',color='gray',markersize=2)
pyplot.plot(shore.mainland[:,0],shore.mainland[:,1],'+',color='green',markersize=3)
#pyplot.plot(outer[:,0],outer[:,1],'o',color='blue',markersize=3)
pyplot.gca().axis('equal')

oFile = os.path.basename(sys.argv[0]).replace('.py','')
pyplot.savefig(oFile+'.svg')

nodes3 = numpy.zeros((points.shape[0],3))
nodes3[:,0] = points[:,0]
nodes3[:,1] = points[:,1]
nodes3[:,2] = 1.

#meshio.write_points_cells(oFile+'.vtk',nodes3,[("triangle", cells)],file_format="vtk")
#CALL = "gmsh -2 {:s}.vtk -o {:s}.msh -format msh2".format(oFile,oFile)
#print('os.system({:s})'.format(CALL))
#os.system(CALL)
