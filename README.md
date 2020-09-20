oceanmesh: Automatic coastal ocean mesh generation
=====================================================
:ocean: :cyclone:

[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=svg)](https://circleci.com/gh/CHLNDDEV/oceanmesh)
[![CodeCov](https://codecov.io/gh/CHLNDDEV/oceanmesh/branch/master/graph/badge.svg)](https://codecov.io/gh/CHLNDDEV/oceanmesh)



Coastal ocean mesh generation from ESRI Shapefiles and digitial elevation models.


Functionality
=============

* A self-contained mesh generation toolkit for the development of coastal oceans meshes composed of two-dimensional, unstructured triangles. The software embeds the mesh generation into an object-orientated framework that integrates the necessary pre- and post-processing tools to inevitably perform numerical simulations with the mesh.
    * The ability to automatically handle and convert arbitrarily complex vector data that represent complex coastal boundaries into practical mesh boundaries used in the mesh generation process.
    * A variety of commonly used mesh size functions that can easily be controlled via name-value pairs.
    * Post-processing and visualization methods to ensure the mesh is ready for simulation.
    * A modified version of the DistMesh mesh generator that can mesh according to geophysical datasets.

Installation
============

For installation, oceanmesh needs [CGAL](https://www.cgal.org/) and
[pybind11](https://github.com/pybind/pybind11):

    sudo apt install libcgal-dev python3-pybind11

After that, clone the repo and oceanmesh can be updated/installed using pip.

    pip install -U -e .

:warning:

**WARNING: THIS PROGRAM IN ACTIVE DEVELOPMENT AND INSTALLATION IS ONLY RECOMMENDED FOR DEVELOPERS AT THIS TIME. WHEN A STABLE API IS REACHED, THE PROGRAM WILL BE AVAILABLE VIA pypi**

Examples
==========

* WIP

Testing
============

To run the `oceanmesh` unit tests (and turn off plots), check out this repository and type
```
MPLBACKEND=Agg pytest --maxfail=1
```


License
=======

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
