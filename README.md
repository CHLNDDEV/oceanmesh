oceanmesh: Automatic coastal ocean mesh generation
=====================================================
:ocean: :cyclone:

[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=svg)](https://circleci.com/gh/CHLNDDEV/oceanmesh)
[![CodeCov](https://codecov.io/gh/CHLNDDEV/oceanmesh/branch/master/graph/badge.svg)](https://codecov.io/gh/CHLNDDEV/oceanmesh)



Coastal ocean mesh generation from ESRI Shapefiles and digitial elevation models.


Functionality
=============


* A toolkit for the development of meshes and their auxiliary files that are used in the simulation of coastal ocean circulation. The software integrates mesh generation with geophysical datasets such as topobathymetric rasters/digital elevation models and shapefiles representing coastal features. It provides some necessary pre- and post-processing tools to inevitably perform a succesful numerical simulations with the developed model.
    * Automatically deal with arbitrarily complex shoreline vector datasets that represent complex coastal boundaries and incorporate the data in an automatic-sense into the mesh generation process.
    * A variety of commonly used mesh size functions to distribute element sizes that can easily be controlled via a simple scripting application interface.
    * Mesh checking and clean-up methods to avoid simulation problems.

Questions?
============

Besides posting issues with the code on Github, you can also ask questions via our Slack channel [here](https://join.slack.com/t/oceanmesh2d/shared_invite/zt-hcu2nag7-NUBw52cxxlYupLrc1hqvhw).

Otherwise please reach out to either Dr. William Pringle (wpringle@nd.edu) or Dr. Keith Roberts (krober@usp.br) with questions or concerns!

Installation
============

For installation, oceanmesh needs [CGAL](https://www.cgal.org/) and
[pybind11](https://github.com/pybind/pybind11):

    sudo apt install libcgal-dev python3-pybind11

After that, clone the repo and oceanmesh can be updated/installed using pip.

    pip install -U -e .

:warning:

**WARNING: THIS PROGRAM IS IN ACTIVE DEVELOPMENT. INSTALLATION IS ONLY RECOMMENDED FOR DEVELOPERS AT THIS TIME. WHEN A STABLE API IS REACHED, THE PROGRAM WILL BE AVAILABLE VIA pypi**

Examples
==========

Build a simple mesh around New York witha minimum element size of 1 km expanding linear from the shoreline to a maximum size of 5 km.


**Here we use the GSHHS shoreline [here](http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip) and the Python package `meshio` to write the mesh to a VTK file for visualization in ParaView.**

![NewYorkMesh](https://user-images.githubusercontent.com/18619644/102819581-7587b600-43b2-11eb-9410-fbf3cadf95b9.png)


```python
import meshio

from oceanmesh import (
    Shoreline,
    distance_sizing_function,
    signed_distance_function,
    generate_mesh,
    make_mesh_boundaries_traversable,
    delete_faces_connected_to_one_face,
)


fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

bbox, min_edge_length = (-75.000, -70.001, 40.0001, 41.9000), 1e3

shore = Shoreline(fname, bbox, min_edge_length)

edge_length = distance_sizing_function(shore, max_edge_length=5e3)

domain = signed_distance_function(shore)

points, cells = generate_mesh(domain, edge_length)

# remove degenerate mesh faces and other common problems in the mesh
points, cells = make_mesh_boundaries_traversable(points, cells)

points, cells = delete_faces_connected_to_one_face(points, cells)

meshio.write_points_cells(
    "simple_new_york.vtk",
    points,
    [("triangle", cells)],
    file_format="vtk",
)
```

Testing
============

To run the `oceanmesh` unit tests (and turn off plots), check out this repository and type
```
MPLBACKEND=Agg pytest --maxfail=1
```


License
=======

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
