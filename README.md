oceanmesh: Automatic coastal ocean mesh generation
=====================================================
:ocean: :cyclone:

[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=svg)](https://circleci.com/gh/CHLNDDEV/oceanmesh)
[![CodeCov](https://codecov.io/gh/CHLNDDEV/oceanmesh/branch/master/graph/badge.svg)](https://codecov.io/gh/CHLNDDEV/oceanmesh)

Coastal ocean mesh generation from ESRI Shapefiles and digital elevation models.

Table of contents
=================

<!--ts-->
   * [oceanmesh](#oceanmesh)
   * [Table of contents](#table-of-contents)
   * [Functionality](#functionality)
   * [Citing](#citing)
   * [Questions or problems](#questions-or-problems)
   * [Installation](#installation)
   * [Examples](#examples)
     * [Reading in geophysical data](#reading-in-geophysical-data)
     * [Building mesh sizing functions](#building-mesh-sizing-functions)
     * [Mesh generation](#mesh-generation)
     * [Multiscale mesh generation](#multiscale-mesh-generation)
     * [Cleaning up the mesh](#cleaning-up-the-mesh)
   * [Testing](#testing)
   * [License](#license)
<!--te-->


Functionality
=============

* A Python package for the development of unstructured triangular meshes that are used in the simulation of coastal ocean circulation. The software integrates mesh generation directly with geophysical datasets such as topobathymetric rasters/digital elevation models and shapefiles representing coastal features. It provides some necessary pre- and post-processing tools to inevitably perform a successful numerical simulation with the developed model.
    * Automatically deal with arbitrarily complex shoreline vector datasets that represent complex coastal boundaries and incorporate the data in an automatic-sense into the mesh generation process.
    * A variety of commonly used mesh size functions to distribute element sizes that can easily be controlled via a simple scripting application interface.
    * Mesh checking and clean-up methods to avoid simulation problems.


Citing
======

The Python version of the algorithm does not yet have a citation; however, similar algorithms and ideas are shared between both version.

```
[1] - Roberts, K. J., Pringle, W. J., and Westerink, J. J., 2019.
      OceanMesh2D 1.0: MATLAB-based software for two-dimensional unstructured mesh generation in coastal ocean modeling,
      Geoscientific Model Development, 12, 1847-1868. https://doi.org/10.5194/gmd-12-1847-2019.
```

Questions or problems
======================

Besides posting issues with the code on Github, you can also ask questions via our Slack channel [here](https://join.slack.com/t/oceanmesh2d/shared_invite/zt-su1q3lh3-C_j6AIOQPrewqZnanhzN7g).

Otherwise please reach out to Dr. Keith Roberts (keithrbt0@gmail.com) with questions or concerns!

Please include version information when posting bug reports.
`oceanmesh` uses [versioneer](https://github.com/python-versioneer/python-versioneer).

The version can be inspected through Python
```
python -c "import oceanmesh; print(oceanmesh.__version__)"
```
or through
```
python setup.py version
```
in the working directory.

To see what's going on with `oceanmesh` you can turn on logging, which is by default suppressed.

```
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
```

Installation
============

For installation, oceanmesh needs [cmake](https://cmake.org/), [CGAL](https://www.cgal.org/):

    sudo apt install cmake libcgal-dev

CGAL and can also be installed with [`conda`](https://www.anaconda.com/products/individual):

    conda install -c conda-forge cgal

After that, clone the repo and oceanmesh can be updated/installed using pip.

    pip install -U -e .

On some clusters/HPC in order to install CGAL, you may need to load/install [gmp](https://gmplib.org/) and [mpfr](https://www.mpfr.org/).
For example, to install:

    sudo apt install libmpfr-dev libgmp3-dev

:warning:

**WARNING: THIS PROGRAM IS IN ACTIVE DEVELOPMENT. INSTALLATION IS ONLY RECOMMENDED FOR DEVELOPERS AT THIS TIME. WHEN A STABLE API IS REACHED, THE PROGRAM WILL BE AVAILABLE VIA pypi**

Examples
==========

Reading in geophysical data
---------------------------
`oceanmesh` uses shoreline vector datasets (i.e., ESRI shapefiles) and digitial elevation models (DEMs) to construct mesh size functions and signed distance functions to adapt mesh resolution for complex and irregularly shaped coastal ocean domains.

Shoreline datasets are necessary to build signed distance functions which define the meshing domain. Here we show how to download a world shoreline dataset referred to as [GSHHG](https://www.ngdc.noaa.gov/mgg/shorelines/) and read it into to `oceanmesh`.

```python
import zipfile

import requests

import oceanmesh as om

# Download and load the GSHHS shoreline
url = "http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip"
filename = url.split("/")[-1]
with open(filename, "wb") as f:
    r = requests.get(url)
    f.write(r.content)

with zipfile.ZipFile("gshhg-shp-2.3.7.zip", "r") as zip_ref:
    zip_ref.extractall("gshhg-shp-2.3.7")

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG code for WGS84 which is what you want to mesh in
# Specify and extent to read in and a minimum mesh size in the unit of the projection
extent = om.Region(extent=(-75.000, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # In the units of the projection!
shoreline = om.Shoreline(
    fname, extent.bbox, min_edge_length, crs=EPSG
)  # NB: the Shoreline class assumes WGS84:4326 if not specified
shoreline.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="shoreline boundaries",
)
# Using our shoreline, we create a signed distance function
# which will be used for meshing later on.
sdf = om.signed_distance_function(shoreline)
```
![Figure_1](https://user-images.githubusercontent.com/18619644/133544070-2d0f2552-c29a-4c44-b0aa-d3649541af4d.png)

DEMs are used to build some mesh size functions (e.g., wavelength, enforcing size bounds, enforce maximum Courant bounds) but are not essential for mesh generation purposes. The DEM used below 'Eastcoast.nc' was created using the Python package [elevation](https://github.com/bopen/elevation) with the following command:
<!--pytest-codeblocks:skip-->
```
eio clip - o EastCoast.nc --bounds -74.4 40.2 -73.4 41.2
```
This data is a clip from the [SRTM30 m](https://lpdaac.usgs.gov/products/srtmgl1nv003/) elevation dataset.

```python
import oceanmesh as om

fdem = "datasets/EastCoast.nc"

# Digital Elevation Models (DEM) can be read into oceanmesh in
# either the NetCDF format or GeoTiff format provided they are
# in geographic coordinates (WGS84)

# If no extents are passed (i.e., the kwarg bbox), then the entire extent of the
# DEM is read into memory.
EPSG = 4326
dem = om.DEM(
    fdem, crs=EPSG
)
dem.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="SRTM 30m",
    cbarlabel="elevation (meters)",
    vmin=-10,  # minimum elevation value in plot
    vmax=10,  # maximum elevation value in plot
)
```
![Figure_2](https://user-images.githubusercontent.com/18619644/133544110-44497a6b-4a5a-482c-9447-cdc2f3663f17.png)

Building mesh sizing functions
------------------------------
In `oceanmesh` mesh resolution can be controlled according to a variety of feature-driven geometric and topo-bathymetric functions. In this section, we briefly explain the major functions and present examples and code. Reasonable values for some of these mesh sizing functions and their affect on the numerical simulation of barotropic tides was investigated in [Roberts et. al, 2019](https://www.sciencedirect.com/science/article/abs/pii/S1463500319301222)

All mesh size functions are defined on regular Caretesian grids. The properties of these grids are abstracted by the [Grid](https://github.com/CHLNDDEV/oceanmesh/blob/40baeeae313eb8ef285acc395c671c36c1b9605f/oceanmesh/grid.py#L33) class.

### Distance and feature size

A high degree of mesh refinement is often necessary near the shoreline boundary to capture its geometric complexity. If mesh resolution is poorly distributed, critical conveyances may be missed, leading to larger-scale errors in the nearshore circulation. Thus, a mesh size function that is equal to a user-defined minimum mesh size h0 along the shoreline boundary, growing as a linear function of the signed distance d from it, may be appropriate.

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
edge_length = om.distance_sizing_function(shoreline, rate=0.15)
ax = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Distance sizing function",
    cbarlabel="mesh size (degrees)",
    hold=True,
)
shoreline.plot(ax=ax)
```
![Figure_3](https://user-images.githubusercontent.com/18619644/133544111-314cb668-7fd2-45db-b754-4dc204617628.png)

One major drawback of the distance mesh size function is that the minimum mesh size will be placed evenly along straight stretches of shoreline. If the distance mesh size function generates too many vertices (or your application can tolerate it), a feature mesh size function that places resolution according to the geometric width of the shoreline should be employed instead ([Conroy et al., 2012](https://link.springer.com/article/10.1007/s10236-012-0574-0);[Koko, 2015](https://ideas.repec.org/a/eee/apmaco/v250y2015icp650-664.html)).

In this function, the feature size (e.g., the width of channels and/or tributaries and the radius of curvature of the shoreline) along the coast is estimated by computing distances to the medial axis of the shoreline geometry. In `oceanmesh`, we have implemented an approximate medial axis method closely following [Koko, (2015)](https://ideas.repec.org/a/eee/apmaco/v250y2015icp650-664.html).

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
sdf = om.signed_distance_function(shoreline)
# Visualize the medial points
edge_length = om.feature_sizing_function(
    shoreline, sdf, max_edge_length=0.05, plot=True
)
ax = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function",
    cbarlabel="mesh size (degrees)",
    hold=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
```
![Figure_4](https://user-images.githubusercontent.com/18619644/133544112-d5fde284-6839-4e45-901d-c81bca9b8000.png)

### Enforcing mesh size gradation

Some mesh size functions will not produce smooth element size transitions when meshed and this can lead to problems with numerical simulation. All mesh size function can thus be graded such that neighboring mesh sizes are bounded above by a constant. Mesh grading edits coarser regions and preserves the finer mesh resolution zones.

Repeating the above but applying a gradation rate of 15% produces the following:
```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
sdf = om.signed_distance_function(shoreline)
edge_length = om.feature_sizing_function(shoreline, sdf, max_edge_length=0.05)
edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.15)
ax = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function with gradation bound",
    cbarlabel="mesh size (degrees)",
    hold=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
```
![Figure_5](https://user-images.githubusercontent.com/18619644/133544114-cedc0750-b33a-4b7c-9fa5-d14b4e169c40.png)

### Wavelength-to-gridscale

In shallow water theory, the wave celerity, and hence the wavelength λ, is proportional to the square root of the depth of the water column. This relationship indicates that more mesh resolution at shallower depths is required to resolve waves that are shorter than those in deep water. With this considered, a mesh size function hwl that ensures a certain number of elements are present per wavelength (usually of the M2-dominant semi-diurnal tidal species) can be deduced.

In this snippet, as before we compute the feature size function, but now we also compute the wavelength-to-gridscale sizing function using the SRTM dataset and compute the minimum of all the functions before grading. We discretize the wavelength of the M2 by 30 elements (e.g., wl=30)
```python
import oceanmesh as om

fdem = "datasets/EastCoast.nc"
fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

min_edge_length = 0.01

dem = om.DEM(fdem, crs=4326)
shoreline = om.Shoreline(fname, dem.bbox, min_edge_length)
sdf = om.signed_distance_function(shoreline)
edge_length1 = om.feature_sizing_function(shoreline, sdf, max_edge_length=0.05)
edge_length2 = om.wavelength_sizing_function(dem, wl=100)
# Compute the minimum of the sizing functions
edge_length = om.compute_minimum([edge_length1, edge_length2])
edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.15)
ax = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function + wavelength + gradation bound",
    cbarlabel="mesh size (degrees)",
    hold=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
```
![Figure_7](https://user-images.githubusercontent.com/18619644/133544116-ba0f9404-a01e-4b30-bb0d-841c8f61224d.png)


### Resolving bathymetric gradients


Cleaning up the mesh
--------------------

After mesh generation has terminated, a secondary round of mesh improvement strategies is applied that is focused towards improving the geometrically worst-quality triangles that often occur near the boundary of the mesh and can make simulation impossible. Low-quality triangles can occur near the mesh boundary because the geospatial datasets used may contain features that have horizontal length scales smaller than the minimum mesh resolution. To handle this issue, a set of algorithms is applied that iteratively addresses the vertex connectivity problems. The application of the following mesh improvement strategies results in a simplified mesh boundary that conforms to the user-requested minimum element size.

Topological defects in the mesh can be removed by ensuring that it is valid, defined as having the following properties:

1. the vertices of each triangle are arranged in counterclockwise order;

2. conformity (a triangle is not allowed to have a vertex of another triangle in its interior); and

3. traversability (the number of boundary segments is equal to the number of boundary vertices, which guarantees a unique path along the mesh boundary).

Here are some of the relevant codes to address these common problems.
<!--pytest-codeblocks:skip-->
```python
# Address (1) above.
points, cells, jx = fix_mesh(points, cells)
# Addresses (2)-(3) above. Remove degenerate mesh faces and other common problems in the mesh
points, cells = make_mesh_boundaries_traversable(points, cells)
# Remove elements (i.e., "faces") connected to only one channel
# These typically occur in channels at or near the grid scale.
points, cells = delete_faces_connected_to_one_face(points, cells)
# Remove low quality boundary elements less than min_qual
points, cells = delete_boundary_faces(points, cells, min_qual=0.15)
# Apply a Laplacian smoother that preserves the element density
points, cells = laplacian2(points, cells)
```


Mesh generation
----------------
Mesh generation is based on the [DistMesh algorithm](http://persson.berkeley.edu/distmesh/) and requires only a signed distance function and a mesh sizing function. These two functions can be defined through the previously elaborated commands above; however, they can also be straightforward functions that take an array of point coordinates and return the signed distance/desired mesh size.

In this example, we demonstrate all of the above to build a mesh around New York, United States with an approximate minimum element size of around 1 km expanding linear with distance from the shoreline to an approximate maximum element size of 5 km.

**Here we use the GSHHS shoreline [here](http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip) and the Python package `meshio` to write the mesh to a VTK file for visualization in ParaView. Other mesh formats are possible; see `meshio` for more details**

```python
import meshio
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

EPSG = 4326  # EPSG:4326 otherwise known as WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection

shore = om.Shoreline(fname, extent.bbox, min_edge_length)

edge_length = om.distance_sizing_function(shore, max_edge_length=0.05)

domain = om.signed_distance_function(shore)

points, cells = om.generate_mesh(domain, edge_length)

# remove degenerate mesh faces and other common problems in the mesh
points, cells = om.make_mesh_boundaries_traversable(points, cells)

points, cells = om.delete_faces_connected_to_one_face(points, cells)

# remove low quality boundary elements less than 15%
points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)

# apply a Laplacian smoother
points, cells = om.laplacian2(points, cells)

# write the mesh with meshio
meshio.write_points_cells(
    "new_york.vtk",
    points,
    [("triangle", cells)],
    file_format="vtk",
)
```

![new_york](https://user-images.githubusercontent.com/18619644/132709756-1759ef99-f810-4edc-9710-66226e851a50.png)

Multiscale mesh generation
---------------------------

The major downside of the DistMesh algorithm is that it cannot handle regional domains with fine mesh refinement or variable datasets due to the intense memory requirements. The multiscale mesh generation technique addresses these problems and enables an arbitrary number of refinment zones to be incorporated seamlessy into the domain.

Areas of finer refinement can be incorporated seamlessly by using the `generate_multiscale_mesh` function. In this case, the user passes lists of signed distance and edge length functions to the mesh generator but besides this the user API remains the same to the previous mesh generation example. The mesh sizing transitions between nests are handled automatically to produce meshes suitable for FEM and FVM numerical simulations through the parameters prefixed with "blend".

```python
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent1 = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length1 = 0.01  # minimum mesh size in domain in projection
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
min_edge_length2 = 4.6e-4  # minimum mesh size in domain in projection
s1 = om.Shoreline(fname, extent1.bbox, min_edge_length1)
sdf1 = om.signed_distance_function(s1)
el1 = om.distance_sizing_function(s1, max_edge_length=0.05)
s2 = om.Shoreline(fname, extent2.bbox, min_edge_length2)
sdf2 = om.signed_distance_function(s2)
el2 = om.distance_sizing_function(s2)
# Control the element size transition
# from coarse to fine with the kwargs prefixed with `blend`
points, cells = om.generate_multiscale_mesh(
    [sdf1, sdf2],
    [el1, el2],
)
# remove degenerate mesh faces and other common problems in the mesh
points, cells = om.make_mesh_boundaries_traversable(points, cells)
# remove singly connected elements (elements connected to only one other element)
points, cells = om.delete_faces_connected_to_one_face(points, cells)
# remove poor boundary elements with quality < 15%
points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
# apply a Laplacian smoother that preservers the mesh size distribution
points, cells = om.laplacian2(points, cells)

# plot it showing the different levels of resolution
triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
gs = gridspec.GridSpec(2, 1)
gs.update(wspace=0.1)
plt.figure()

ax = plt.subplot(gs[0, 0])  #
ax.set_aspect("equal")
ax.triplot(triang, "-", lw=0.5)
ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")

ax = plt.subplot(gs[1, 0])  #
buf = 0.07
ax.set_xlim([min(bbox2[:,0])-buf,max(bbox2[:,0])+buf])
ax.set_ylim([min(bbox2[:,1])-buf,max(bbox2[:,1])+buf])
ax.set_aspect("equal")
ax.triplot(triang, "-", lw=0.5)
ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")

plt.show()
```
<img width="747" alt="image" src="https://user-images.githubusercontent.com/21131934/136140049-9eee309a-987f-4128-9fe2-bb207f972be3.png">

See the tests inside the `testing/` folder for more inspiration. Work is ongoing on this package.

Testing
============

To run the `oceanmesh` unit tests (and turn off plots), check out this repository and type `tox`. `tox` can be installed via pip.

License
=======

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
