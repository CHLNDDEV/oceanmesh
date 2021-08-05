import os,sys,time
import matplotlib.pyplot as pyplot
import numpy as np
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
import shapefile, shapely.geometry

shp='ocean.shp'
with shapefile.Reader(shp) as shpf:
  shapes = shpf.shapes()
  n = len(shapes)
  for shape in shapes:
    if n>1:
      print('WARN, {:d} polygons in \'{:s}\'. Continue with item[0].'.format(n,os.path.basename(shp)))
      break

  outer = np.asarray(shape.points)
  bbox = (outer[:,0].min(),outer[:,0].max(),outer[:,1].min(),outer[:,1].max())
  outer = np.append(outer, [[np.nan, np.nan]], axis=0)
  print('test_circ::',shp,bbox)

del(shapes,shape,shpf)

shp='land.shp'

min_edge_length = 1.e3
max_edge_length = 5.e3

shore = Shoreline(shp, bbox, min_edge_length)
shore.outer = outer
#print(dir(shore))

#aise RuntimeError('HALT')


edge_length = distance_sizing_function(shore,rate=0.10, max_edge_length=max_edge_length)
domain = signed_distance_function(shore)

points, cells = generate_mesh(domain, edge_length)

pyplot.figure(1)
pyplot.clf()
#pyplot.triplot(points[:,0],points[:,1],cells,'.-',lw=0.5,color='0.5')
pyplot.plot(shore.boubox[:,0],shore.boubox[:,1],'x',color='r',markersize=5)
pyplot.plot(shore.inner[:,0],shore.inner[:,1],'.',color='gray',markersize=3)
pyplot.plot(shore.mainland[:,0],shore.mainland[:,1],'+',color='green',markersize=3)
#pyplot.plot(outer[:,0],outer[:,1],'o',color='blue',markersize=3)
pyplot.gca().axis('equal')

oFile = os.path.basename(sys.argv[0]).replace('.py','')
pyplot.savefig(oFile+'.svg')

nodes3 = np.zeros((points.shape[0],3))
nodes3[:,0] = points[:,0]
nodes3[:,1] = points[:,1]
nodes3[:,2] = 1.

#meshio.write_points_cells(oFile+'.vtk',nodes3,[("triangle", cells)],file_format="vtk")
#CALL = "gmsh -2 {:s}.vtk -o {:s}.msh -format msh2".format(oFile,oFile)
#print('os.system({:s})'.format(CALL))
#os.system(CALL)
