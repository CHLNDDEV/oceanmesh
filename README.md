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
