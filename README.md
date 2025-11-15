# oceanmesh: Automatic coastal ocean mesh generation

:ocean: :cyclone:

[![Tests](https://github.com/CHLNDDEV/oceanmesh/actions/workflows/testing.yml/badge.svg)](https://github.com/CHLNDDEV/oceanmesh/actions/workflows/testing.yml)

[![CodeCov](https://codecov.io/gh/CHLNDDEV/oceanmesh/branch/master/graph/badge.svg)](https://codecov.io/gh/CHLNDDEV/oceanmesh)

Coastal ocean mesh generation from vector and raster GIS data.

# Table of contents

<!--ts-->

- [oceanmesh](#oceanmesh)
- [Table of contents](#table-of-contents)
- [Functionality](#functionality)
- [Citing](#citing)
- [Questions or problems](#questions-or-problems)
- [Installation](#installation)
- [Examples](#examples)
  - [Setting the region](#setting-the-region)
  - [Reading in geophysical data](#reading-in-geophysical-data)
  - [Building mesh sizing functions](#building-mesh-sizing-functions)
  - [Mesh generation](#mesh-generation)
  - [Multiscale mesh generation](#multiscale-mesh-generation)
    - [Global mesh generation with regional refinement](#global-mesh-generation-with-regional-refinement)
  - [Cleaning up the mesh](#cleaning-up-the-mesh)
- [Testing](#testing)
- [License](#license)

<!--te-->

# Functionality

- A Python package for the development of unstructured triangular meshes that are used in the simulation of coastal ocean circulation. The software integrates mesh generation directly with geophysical datasets such as topo-bathymetric rasters/digital elevation models and shapefiles representing coastal features. It provides some necessary pre- and post-processing tools to inevitably perform a successful numerical simulation with the developed model.

  - Automatically deal with arbitrarily complex shoreline vector datasets that represent complex coastal boundaries and incorporate the data in an automatic-sense into the mesh generation process.
  - A variety of commonly used mesh size functions to distribute element sizes that can easily be controlled via a simple scripting application interface.
  - Mesh checking and clean-up methods to avoid simulation problems.

  # Third-Party Code

  OceanMesh includes vendored copies of the following third-party libraries:

  - inpoly-python by Darren Engwirda — fast point-in-polygon testing
    - Source: https://github.com/dengwirda/inpoly-python
    - License: Custom (see `oceanmesh/_vendor/inpoly/LICENSE_INPOLY.txt`)
    - Vendored to avoid build dependency issues with Cython extensions; we use the
      pure-Python implementation for maximum compatibility.
    - Performance: The pure-Python implementation is adequate for typical mesh
      generation workloads, but for very large point-in-polygon queries it can be
      slower than the Cython-accelerated kernel.
    - Optional acceleration: If you have a compiled extension available locally,
      you can enable it by setting the environment variable
      `PYTHON_INPOLY_ACCEL=1` before running. OceanMesh will attempt to use a
      compatible `inpoly_` kernel when this flag is set, otherwise it will
      default to the pure-Python path.
    - Tips for large inputs: Consider batching very large point query arrays in
      your own workflows to reduce peak memory use when evaluating polygons.

### Performance Optimization

The vendored `inpoly` package includes an optional Cython-compiled extension that can provide large speedups (often 60–115×) for point-in-polygon queries. By default, oceanmesh uses the pure-Python implementation so it works everywhere without compilers.

Enable the fast Cython kernel:

1. Install with the optional performance extras (editable or source installs):

```bash
pip install -e .[fast]
```

2. Build from source (force compilation):

```bash
pip install cython numpy
pip install --no-binary oceanmesh .
```

3. From a cloned repo (editable dev mode):

<!--pytest-codeblocks:skip-->

```bash
git clone https://github.com/CHLNDDEV/oceanmesh.git
cd oceanmesh
pip install -e .[fast]
```

Check which implementation is active:

```python
import oceanmesh._vendor.inpoly.inpoly2 as inpoly_mod

print("Compiled kernel:", getattr(inpoly_mod, "_COMPILED_KERNEL_AVAILABLE", False))
```

Note: The Cython extension is optional and marked as such in setup. If compilation fails or a compiler is unavailable, installation still succeeds and the pure-Python fallback is used automatically.

# Citing

The Python version of the algorithm does not have a citation; however, similar algorithms and ideas are shared between the published MATLAB version.

```
[1] - Roberts, K. J., Pringle, W. J., and Westerink, J. J., 2019.
      OceanMesh2D 1.0: MATLAB-based software for two-dimensional unstructured mesh generation in coastal ocean modeling,
      Geoscientific Model Development, 12, 1847-1868. https://doi.org/10.5194/gmd-12-1847-2019.
```

# Questions or problems

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

To see what's going on with `oceanmesh` when running scripts, you can turn on logging (which is by default suppressed) by importing the two modules `sys` and `logging` and then placing one of the three following logging commands after your imports in your calling script. The amount of information returned is the greatest with `logging.DEBUG` and leas with `logging.WARNING`.

```
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
```

# Installation

The notes below refer to installation on platforms other than MS Windows. For Windows, refer to the following section.

For installation, `oceanmesh` needs [cmake](https://cmake.org/), [CGAL](https://www.cgal.org/):

```
sudo apt install cmake libcgal-dev
```

CGAL and can also be installed with [`conda`](https://www.anaconda.com/products/individual):

```
conda install -c conda-forge cgal
```

After that, clone the repo and `oceanmesh` can be updated/installed using pip.

```
pip install -U -e .
```

On some clusters/HPC in order to install CGAL, you may need to load/install [gmp](https://gmplib.org/) and [mpfr](https://www.mpfr.org/).
For example, to install:

```
sudo apt install libmpfr-dev libgmp3-dev
```

# Installation on Windows

Python under Windows can easily experience DLL hell due to version incompatibilities. Such is the case for the combination of support packages required for OceanMesh. For this reason, we have provided 'install_cgal.bat' to build a CGAL development distribution separately as a prerequisite.

Prerequisites to build CGAL using the provided batch file are as follows:

- Windows 10 or later
- Visual Studio with C++
- CMake
- Git

After successful intallation of a CGAL development package, proceed via one of the two options below to generate a python environment with OceanMesh installed.

If you are using a conda-based Python distribution, then 'install_oceanmesh.bat' should take care of everything, provided no package conflicts arise.

If you have a different Python distribution, or if you do not want to use packages from conda forge, then the following process may be required:

1. Obtain binary wheels for your python distribution for the latest GDAL, Fiona, and Rasterio (https://www.lfd.uci.edu/~gohlke/pythonlibs).
1. Create a new virtual environment and activate it.
1. Execute: cmd.exe /C "for %f in (GDAL\*.whl Fiona\*.whl rasterio\*.whl) do pip install %f"
1. Execute: pip install geopandas rasterio scikit-fmm pybind11
1. Execute: python setup.py install

:warning:

**WARNING: THIS PROGRAM IS IN ACTIVE DEVELOPMENT. INSTALLATION IS ONLY RECOMMENDED FOR DEVELOPERS AT THIS TIME. WHEN A STABLE API IS REACHED, THE PROGRAM WILL BE AVAILABLE VIA pypi**

# Examples

## Setting the region

`oceanmesh` can mesh a polygonal region in the vast majority of coordinate reference systems (CRS). Thus, all `oceanmesh` scripts begin with declaring the extent and CRS to be used and/or transofmring to a different CRS like this.

```python
import oceanmesh as om

EPSG = 32619  # A Python int, dict, or str containing the CRS information (in this case UTM19N)
bbox = (
    -70.29637,
    -43.56508,
    -69.65537,
    43.88338,
)  # the extent of the domain (can also be a multi-polygon delimited by rows of np.nan)
extent = om.Region(
    extent=bbox, crs=4326
)  # set the region (the bbox is given here in EPSG:4326 or WGS84)
extent = extent.transform_to(EPSG)  # Now I transform to the desired EPSG (UTM19N)
print(
    extent.bbox
)  # now the extents are in the desired CRS and can be passed to various functions later on
```

## Reading in geophysical data

`oceanmesh` uses shoreline vector datasets (i.e., ESRI shapefiles) and digital elevation models (DEMs) to construct mesh size functions and signed distance functions to adapt mesh resolution for complex and irregularly shaped coastal ocean domains.

Shoreline datasets are necessary to build signed distance functions which define the meshing domain. Here we show how to download a world shoreline dataset referred to as [GSHHG](https://www.ngdc.noaa.gov/mgg/shorelines/) and read it into to `oceanmesh`.

**Note**: The following code blocks demonstrate downloading and using the GSHHS shoreline dataset. If you're running these examples locally, execute the download code block first. Subsequent examples in this README assume the dataset has been downloaded to your working directory. In CI, individual code blocks run in isolation, so blocks that depend on downloaded data are marked to be skipped.

<!--pytest-codeblocks:skip-->

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
EPSG = 4326  # EPSG code for WGS84
# Specify an extent to read in and a minimum mesh size in the units of the projection
extent = om.Region(extent=(-75.000, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01

# Preferred: pass the Region object directly (automatically uses the Region's CRS)
shoreline = om.Shoreline(fname, extent, min_edge_length)

# Alternative (backward compatible): pass bbox and crs separately
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)
shoreline.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="shoreline boundaries",
)
# Using our shoreline, we create a signed distance function
# which will be used for meshing later on.
```

### Working with Projected Coordinate Systems

When working in projected CRSs (e.g., UTM), always prefer passing a Region object so that both the bbox and CRS travel together.

```python
import oceanmesh as om

# Working with UTM coordinates (e.g., EPSG:32610 - UTM Zone 10N)
EPSG = 32610
extent = om.Region(extent=(xmin, xmax, ymin, ymax), crs=EPSG)
min_edge_length = 15  # meters (UTM uses meters)

# Correct: Region object carries both bbox and CRS
shore = om.Shoreline(fname, extent, min_edge_length)

# Also correct: Explicit CRS parameter with tuple
shore = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)

# Wrong: bbox in UTM but CRS defaults to WGS84 (do NOT do this)
# shore = om.Shoreline(fname, extent.bbox, min_edge_length)  # Error likely
```

> Note: Best Practice — Always pass a Region object to Shoreline instead of just the bbox. This ensures the CRS is automatically matched and prevents coordinate system mismatches.

## Global mesh generation with regional refinement

Oceanmesh also supports combining a global mesh with one or more regional refinement zones. This pattern is valuable for global ocean circulation models that require higher resolution in coastal or shelf regions (e.g., Australia) while keeping coarser resolution elsewhere. The workflow proceeds in two conceptual steps: (1) define all sizing functions in WGS84 (EPSG:4326) latitude/longitude coordinates; (2) generate the global mesh in a stereographic projection. All coordinate transformations between stereographic space and WGS84 are handled automatically during mesh generation—users only supply domains and sizing functions with the correct ordering and flags.

<!--pytest-codeblocks:skip-->

```python
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.gridspec as gridspec

import oceanmesh as om
from oceanmesh.region import to_lat_lon

# ---------------------------------------------------------------------------
# File paths (shapefiles included with tests)
# global_latlon.shp: coastline in WGS84 for sizing functions
# global_stereo.shp: same coastline already in stereographic projection for meshing
fname_global_latlon = "tests/global/global_latlon.shp"
fname_global_stereo = "tests/global/global_stereo.shp"

EPSG = 4326  # WGS84

# ---------------------------------------------------------------------------
# Global (coarse) domain definition in WGS84
global_bbox = (-180.0, 180.0, -89.0, 90.0)
global_region = om.Region(extent=global_bbox, crs=EPSG)
min_edge_length1 = 1.0   # degrees (approximate target minimum at shoreline; units follow EPSG:4326)
max_edge_length1 = 3.0   # degrees (approximate coarse offshore size)
shoreline_global_latlon = om.Shoreline(fname_global_latlon, global_region, min_edge_length1)  # Preferred: pass Region object to automatically use its CRS
sdf_global_latlon = om.signed_distance_function(shoreline_global_latlon)

> Note: Passing the `Region` directly couples bbox + CRS and is especially important in global/multiscale workflows that mix WGS84 sizing functions with stereographic meshing. See "Working with Projected Coordinate Systems" for details.
<!-- retained above after moving -->

# Sizing functions (distance + feature) built in WGS84 (units: degrees). Internal routines convert to meters where needed.
edge_length_global_dist = om.distance_sizing_function(shoreline_global_latlon, rate=0.11)
edge_length_global_feat = om.feature_sizing_function(
    shoreline_global_latlon,
    sdf_global_latlon,
    max_edge_length=max_edge_length1,
)
edge_length_global = om.compute_minimum([edge_length_global_dist, edge_length_global_feat])
# Apply gradation with stereo awareness (global domain will later mesh in stereographic space)
edge_length_global = om.enforce_mesh_gradation(edge_length_global, gradation=0.15, stereo=True)
# Note: gradation uses stereo=True for global domain

# ---------------------------------------------------------------------------
# Regional (fine) domain: Australia example in WGS84
aus_bbox = (110.0, 160.0, -45.0, -10.0)
aus_region = om.Region(extent=aus_bbox, crs=EPSG)
min_edge_length2 = 0.25  # degrees (refined shoreline resolution; could use projected CRS instead, e.g., UTM)
max_edge_length2 = 1.5   # degrees (regional offshore resolution)
shoreline_regional = om.Shoreline(fname_global_latlon, aus_region, min_edge_length2)  # Preferred: pass Region object to automatically use its CRS
sdf_regional = om.signed_distance_function(shoreline_regional)
edge_length_regional_dist = om.distance_sizing_function(shoreline_regional, rate=0.13)
edge_length_regional_feat = om.feature_sizing_function(
    shoreline_regional,
    sdf_regional,
    max_edge_length=max_edge_length2,
)
edge_length_regional = om.compute_minimum([edge_length_regional_dist, edge_length_regional_feat])
edge_length_regional = om.enforce_mesh_gradation(edge_length_regional, gradation=0.12)
# Regional domain uses standard lat/lon coordinates

# ---------------------------------------------------------------------------
# Stereographic domain for global meshing (must set stereo=True)
shoreline_global_stereo = om.Shoreline(
    fname_global_stereo,
    global_region,
    min_edge_length1,
    stereo=True,
)  # Preferred: pass Region object to automatically use its CRS
sdf_global_stereo = om.signed_distance_function(shoreline_global_stereo)
# Global domain must use stereographic projection for meshing

# ---------------------------------------------------------------------------
# Multiscale mesh generation (global + regional)
# First domain (global) has stereo=True, second (regional) does not.
# Coordinate transformations between projections handled automatically.
points, cells = om.generate_multiscale_mesh(
    [sdf_global_stereo, sdf_regional],
    [edge_length_global, edge_length_regional],
    blend_width=1.0e6,    # transition width (meters). For basin-scale transitions use ~1e5–1e6 m (≈1–9° assuming ~111e3 m per degree).
    blend_max_iter=50,    # iterations for the blending step
    max_iter=75,          # iterations for initial domain meshing
)

# ---------------------------------------------------------------------------
# Standard cleanup operations to improve mesh quality
points, cells = om.make_mesh_boundaries_traversable(points, cells)
points, cells = om.delete_faces_connected_to_one_face(points, cells)
points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
points, cells = om.laplacian2(points, cells, max_iter=50)

# ---------------------------------------------------------------------------
# (Optional) Visualization
# Convert stereographic mesh coordinates back to lat/lon for plotting
lon, lat = to_lat_lon(points[:, 0], points[:, 1])
triang = tri.Triangulation(lon, lat, cells)

# Two-panel figure: global view + zoomed Australia refinement
fig = plt.figure(figsize=(10, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1])

ax0 = fig.add_subplot(gs[0])
ax0.set_title("Global mesh with Australia refinement")
ax0.set_aspect("equal")
ax0.triplot(triang, lw=0.2, color="gray")
ax0.plot([aus_bbox[0], aus_bbox[1], aus_bbox[1], aus_bbox[0], aus_bbox[0]],
         [aus_bbox[2], aus_bbox[2], aus_bbox[3], aus_bbox[3], aus_bbox[2]],
         "r--", lw=1.0, label="Australia bbox")
ax0.set_xlim(-180, 180)
ax0.set_ylim(-90, 90)
ax0.legend(loc="lower left")

ax1 = fig.add_subplot(gs[1])
ax1.set_title("Refined Australia region")
ax1.set_aspect("equal")
ax1.triplot(triang, lw=0.25, color="black")
ax1.set_xlim(aus_bbox[0], aus_bbox[1])
ax1.set_ylim(aus_bbox[2], aus_bbox[3])

plt.tight_layout()
# plt.show()  # Uncomment to display interactively
```

Key points:

- Ordering: the global domain (stereo=True) must be first in the `generate_multiscale_mesh` domain list.
- Projection split: sizing functions are defined in WGS84; meshing of the global extent runs in stereographic space.
- Regional domains: remain in standard EPSG:4326 coordinates and must be contained within the global bbox.
- Regional domains may use a projected CRS (e.g., UTM). Oceanmesh will transform and blend sizing grids across mixed CRSs automatically.
- Automatic handling: oceanmesh transforms query coordinates between stereographic and lat/lon (and projected regional CRSs) as needed—no manual reprojection code is required.
- Validation: `generate_multiscale_mesh` enforces CRS compatibility, bbox containment, and correct stereo flag usage; error messages guide fixes.

![Global Regional Multiscale](path/to/image.png)
*The image above (placeholder) would show the global mesh with a refined Australia region.*
![Figure_1](https://user-images.githubusercontent.com/18619644/133544070-2d0f2552-c29a-4c44-b0aa-d3649541af4d.png)

DEMs are used to build some mesh size functions (e.g., wavelength, enforcing size bounds, enforce maximum Courant bounds) but are not essential for mesh generation purposes. The DEM used below 'Eastcoast.nc' was created using the Python package [elevation](https://github.com/bopen/elevation) with the following command:

<!--pytest-codeblocks:skip-->

```
eio clip - o EastCoast.nc --bounds -74.4 40.2 -73.4 41.2
```

This data is a clip from the [SRTM30 m](https://lpdaac.usgs.gov/products/srtmgl1nv003/) elevation dataset.

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fdem = "datasets/EastCoast.nc"

# Digital Elevation Models (DEM) can be read into oceanmesh in
# either the NetCDF format or GeoTiff format provided they are
# in geographic coordinates (WGS84)

# If no extents are passed (i.e., the kwarg bbox), then the entire extent of the
# DEM is read into memory.
# Note: the DEM will be projected to the desired CRS automatically.
EPSG = 4326
dem = om.DEM(fdem, crs=EPSG)
dem.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="SRTM 30m",
    cbarlabel="elevation (meters)",
    vmin=-10,  # minimum elevation value in plot
    vmax=10,  # maximum elevation value in plot
)
```

<!--pytest-codeblocks:skip-->

![Figure_2](https://user-images.githubusercontent.com/18619644/133544110-44497a6b-4a5a-482c-9447-cdc2f3663f17.png)

# Defining the domain

______________________________________________________________________

The domain is defined by a signed distance function. A signed distance function can be automatically generated from a complex coastal ocean domain as such

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(
    fname, extent, min_edge_length
)  # Preferred: pass Region directly
# Alternative (tuple + explicit CRS):
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)

# build a signed distance functiona automatically
sdf = om.signed_distance_function(shoreline)
```

In some situations, it is necessary to flip the definition of what's `inside` the meshing domain. This can be accomplished via the `invert` kwarg for the `signed_distance_function`.

<!--pytest-codeblocks:skip-->

```python
sdf = om.signed_distance_function(shoreline, invert=True)
```

Setting `invert=True` will be mesh the 'land side' of the domain rather than the ocean.

## Building mesh sizing functions

In `oceanmesh` mesh resolution can be controlled according to a variety of feature-driven geometric and topo-bathymetric functions. In this section, we briefly explain the major functions and present examples and code. Reasonable values for some of these mesh sizing functions and their affect on the numerical simulation of barotropic tides was investigated in [Roberts et. al, 2019](https://www.sciencedirect.com/science/article/abs/pii/S1463500319301222)

All mesh size functions are defined on regular Cartesian grids. The properties of these grids are abstracted by the [Grid](https://github.com/CHLNDDEV/oceanmesh/blob/40baeeae313eb8ef285acc395c671c36c1b9605f/oceanmesh/grid.py#L33) class.

### Distance and feature size

A high degree of mesh refinement is often necessary near the shoreline boundary to capture its geometric complexity. If mesh resolution is poorly distributed, critical conveyances may be missed, leading to larger-scale errors in the nearshore circulation. Thus, a mesh size function that is equal to a user-defined minimum mesh size h0 along the shoreline boundary, growing as a linear function of the signed distance d from it, may be appropriate.

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent, min_edge_length)  # Preferred pattern
# Alternative:
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)
edge_length = om.distance_sizing_function(shoreline, rate=0.15)
fig, ax, pc = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Distance sizing function",
    cbarlabel="mesh size (degrees)",
    holding=True,
)
shoreline.plot(ax=ax)
<!--pytest-codeblocks:skip-->
```

![Figure_3](https://user-images.githubusercontent.com/18619644/133544111-314cb668-7fd2-45db-b754-4dc204617628.png)

One major drawback of the distance mesh size function is that the minimum mesh size will be placed evenly along straight stretches of shoreline. If the distance mesh size function generates too many vertices (or your application can tolerate it), a feature mesh size function that places resolution according to the geometric width of the shoreline should be employed instead ([Conroy et al., 2012](https://link.springer.com/article/10.1007/s10236-012-0574-0);[Koko, 2015](https://ideas.repec.org/a/eee/apmaco/v250y2015icp650-664.html)).

In this function, the feature size (e.g., the width of channels and/or tributaries and the radius of curvature of the shoreline) along the coast is estimated by computing distances to the medial axis of the shoreline geometry. In `oceanmesh`, we have implemented an approximate medial axis method closely following [Koko, (2015)](https://ideas.repec.org/a/eee/apmaco/v250y2015icp650-664.html).

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent, min_edge_length)
# Recommended: Passing Region lets bbox+CRS travel together, avoiding mismatches.
# Alternative:
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)
sdf = om.signed_distance_function(shoreline)
# Visualize the medial points
edge_length = om.feature_sizing_function(
    shoreline, sdf, max_edge_length=0.05, plot=True
)
fig, ax, pc = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function",
    cbarlabel="mesh size (degrees)",
    holding=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
```

![Figure_4](https://user-images.githubusercontent.com/18619644/133544112-d5fde284-6839-4e45-901d-c81bca9b8000.png)

### Enforcing mesh size gradation

Some mesh size functions will not produce smooth element size transitions when meshed and this can lead to problems with numerical simulation. All mesh size function can thus be graded such that neighboring mesh sizes are bounded above by a constant. Mesh grading edits coarser regions and preserves the finer mesh resolution zones.

Repeating the above but applying a gradation rate of 15% produces the following:

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"
EPSG = 4326  # EPSG:4326 or WGS84
extent = om.Region(extent=(-75.00, -70.001, 40.0001, 41.9000), crs=EPSG)
min_edge_length = 0.01  # minimum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent, min_edge_length)
# Alternative:
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)
sdf = om.signed_distance_function(shoreline)
edge_length = om.feature_sizing_function(shoreline, sdf, max_edge_length=0.05)
edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.15)
fig, ax, pc = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function with gradation bound",
    cbarlabel="mesh size (degrees)",
    holding=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
<!--pytest-codeblocks:skip-->
```

![Figure_5](https://user-images.githubusercontent.com/18619644/133544114-cedc0750-b33a-4b7c-9fa5-d14b4e169c40.png)

### Wavelength-to-gridscale

In shallow water theory, the wave celerity, and hence the wavelength λ, is proportional to the square root of the depth of the water column. This relationship indicates that more mesh resolution at shallower depths is required to resolve waves that are shorter than those in deep water. With this considered, a mesh size function hwl that ensures a certain number of elements are present per wavelength (usually of the M2-dominant semi-diurnal tidal species but the frequency of the dominant wave can be specified via kwargs) can be deduced.

In this snippet, as before we compute the feature size function, but now we also compute the wavelength-to-gridscale sizing function using the SRTM dataset and compute the minimum of all the functions before grading. We discretize the wavelength of the M2 by 30 elements (e.g., wl=30)

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fdem = "datasets/EastCoast.nc"
fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

min_edge_length = 0.01

extent = om.Region(extent=(-74.3, -73.8, 40.3,40.8), crs=4326)
dem = om.DEM(fdem, bbox=extent,crs=4326)
shoreline = om.Shoreline(fname, dem.bbox, min_edge_length)  # DEM.bbox already a tuple; Region pattern not applicable here
sdf = om.signed_distance_function(shoreline)
edge_length1 = om.feature_sizing_function(shoreline, sdf, max_edge_length=0.05)
edge_length2 = om.wavelength_sizing_function(
    dem, wl=100, period=12.42 * 3600
)  # use the M2-tide period (in seconds)
# Compute the minimum of the sizing functions
edge_length = om.compute_minimum([edge_length1, edge_length2])
edge_length = om.enforce_mesh_gradation(edge_length, gradation=0.15)
fig, ax, pc = edge_length.plot(
    xlabel="longitude (WGS84 degrees)",
    ylabel="latitude (WGS84 degrees)",
    title="Feature sizing function + wavelength + gradation bound",
    cbarlabel="mesh size (degrees)",
    holding=True,
    xlim=[-74.3, -73.8],
    ylim=[40.3, 40.8],
)
shoreline.plot(ax=ax)
<!--pytest-codeblocks:skip-->
```

![Figure_7](https://user-images.githubusercontent.com/18619644/133544116-ba0f9404-a01e-4b30-bb0d-841c8f61224d.png)

### Resolving bathymetric gradients

The distance, feature size, and/or wavelength mesh size functions can lead to coarse mesh resolution in deeper waters that under-resolve and smooth over the sharp topographic gradients that characterize the continental shelf break. These slope features can be important for coastal ocean models in order to capture dissipative effects driven by the internal tides, transmissional reflection at the shelf break that controls the astronomical tides, and trapped shelf waves. The bathymetry field contains often excessive details that are not relevant for most flows, thus the bathymetry can be smoothed by a variety of filters (e.g., lowpass, bandpass, and highpass filters) before calculating the mesh sizes.

<!--pytest-codeblocks:skip-->

```python
import oceanmesh as om

fdem = "datasets/EastCoast.nc"
fname = "gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp"

EPSG = 4326  # EPSG:4326 or WGS84
bbox = (-74.4, -73.4, 40.2, 41.2)
extent = om.Region(extent=bbox, crs=EPSG)
dem = om.DEM(fdem, crs=EPSG)

min_edge_length = 0.0025  # minimum mesh size in domain in projection
max_edge_length = 0.10  # maximum mesh size in domain in projection
shoreline = om.Shoreline(fname, extent, min_edge_length)
# Alternative:
# shoreline = om.Shoreline(fname, extent.bbox, min_edge_length, crs=EPSG)
sdf = om.signed_distance_function(shoreline)

edge_length1 = om.feature_sizing_function(
    shoreline,
    sdf,
    max_edge_length=max_edge_length,
    crs=EPSG,
)
edge_length2 = om.bathymetric_gradient_sizing_function(
    dem,
    slope_parameter=5.0,
    filter_quotient=50,
    min_edge_length=min_edge_length,
    max_edge_length=max_edge_length,
    crs=EPSG,
)  # will be reactivated
edge_length3 = om.compute_minimum([edge_length1, edge_length2])
edge_length3 = om.enforce_mesh_gradation(edge_length3, gradation=0.15)
```

## Cleaning up the mesh

After mesh generation has terminated, a secondary round of mesh improvement strategies is applied that is focused towards improving the geometrically worst-quality triangles that often occur near the boundary of the mesh and can make simulation impossible. Low-quality triangles can occur near the mesh boundary because the geospatial datasets used may contain features that have horizontal length scales smaller than the minimum mesh resolution. To handle this issue, a set of algorithms is applied that iteratively addresses the vertex connectivity problems. The application of the following mesh improvement strategies results in a simplified mesh boundary that conforms to the user-requested minimum element size.

Topological defects in the mesh can be removed by ensuring that it is valid, defined as having the following properties:

1. the vertices of each triangle are arranged in counterclockwise order;

1. conformity (a triangle is not allowed to have a vertex of another triangle in its interior); and

1. traversability (the number of boundary segments is equal to the number of boundary vertices, which guarantees a unique path along the mesh boundary).

Here are some of the relevant codes to address these common problems.

<!--pytest-codeblocks:skip-->

```python
# Address (1) above.
points, cells = fix_mesh(points, cells)
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

## Mesh generation

Mesh generation is based on the [DistMesh algorithm](http://persson.berkeley.edu/distmesh/) and requires only a signed distance function and a mesh sizing function. These two functions can be defined through the previously elaborated commands above; however, they can also be straightforward functions that take an array of point coordinates and return the signed distance/desired mesh size.

In this example, we demonstrate all of the above to build a mesh around New York, United States with an approximate minimum element size of around 1 km expanding linear with distance from the shoreline to an approximate maximum element size of 5 km.

**Here we use the GSHHS shoreline [here](http://www.soest.hawaii.edu/pwessel/gshhg/gshhg-shp-2.3.7.zip) and the Python package `meshio` to write the mesh to a VTK file for visualization in ParaView. Other mesh formats are possible; see `meshio` for more details**

<!--pytest-codeblocks:skip-->

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

## Multiscale mesh generation

The major downside of the DistMesh algorithm is that it cannot handle regional domains with fine mesh refinement or variable datasets due to the intense memory requirements. The multiscale mesh generation technique addresses these problems and enables an arbitrary number of refinement zones to be incorporated seamlessly into the domain.

Areas of finer refinement can be incorporated seamlessly by using the `generate_multiscale_mesh` function. In this case, the user passes lists of signed distance and edge length functions to the mesh generator but besides this the user API remains the same to the previous mesh generation example. The mesh sizing transitions between nests are handled automatically to produce meshes suitable for FEM and FVM numerical simulations through the parameters prefixed with "blend".

<!--pytest-codeblocks:skip-->

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

# Control the element size transition from coarse to fine
points, cells = om.generate_multiscale_mesh([sdf1, sdf2], [el1, el2])

# Remove some common mesh issues and smooth
points, cells = om.make_mesh_boundaries_traversable(points, cells)
points, cells = om.delete_faces_connected_to_one_face(points, cells)
points, cells = om.delete_boundary_faces(points, cells, min_qual=0.15)
points, cells = om.laplacian2(points, cells)

# Plot showing different levels of resolution
triang = tri.Triangulation(points[:, 0], points[:, 1], cells)
gs = gridspec.GridSpec(2, 2)
gs.update(wspace=0.5)
plt.figure()

bbox3 = np.array(
    [
        [-73.78, 40.60],
        [-73.75, 40.60],
        [-73.75, 40.64],
        [-73.78, 40.64],
        [-73.78, 40.60],
    ],
    dtype=float,
)

ax = plt.subplot(gs[0, 0])
ax.set_aspect("equal")
ax.triplot(triang, "-", lw=1)
ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")
ax.plot(bbox3[:, 0], bbox3[:, 1], "m--")

ax = plt.subplot(gs[0, 1])
ax.set_aspect("equal")
ax.triplot(triang, "-", lw=1)
ax.plot(bbox2[:, 0], bbox2[:, 1], "r--")
ax.set_xlim(np.amin(bbox2[:, 0]), np.amax(bbox2[:, 0]))
ax.set_ylim(np.amin(bbox2[:, 1]), np.amax(bbox2[:, 1]))
ax.plot(bbox3[:, 0], bbox3[:, 1], "m--")

ax = plt.subplot(gs[1, :])
ax.set_aspect("equal")
ax.triplot(triang, "-", lw=1)
ax.set_xlim(-73.78, -73.75)
ax.set_ylim(40.60, 40.64)
plt.show()
```

![Multiscale](https://user-images.githubusercontent.com/18619644/136119785-8746552d-4ff6-44c3-9aa1-3e4981ba3518.png)

## Global mesh generation

Using oceanmesh is now possible for global meshes.
The process is done in two steps:

- first the definition of the sizing functions in WGS84 coordinates,
- then the mesh generation is done in the stereographic projection

<!--pytest-codeblocks:skip-->

```python
import os
import numpy as np
import oceanmesh as om
from oceanmesh.region import to_lat_lon
import matplotlib.pyplot as plt


# utilities functions for plotting
def crosses_dateline(lon1, lon2):
    return abs(lon1 - lon2) > 180


def filter_triangles(points, cells):
    filtered_cells = []
    for cell in cells:
        p1, p2, p3 = points[cell[0]], points[cell[1]], points[cell[2]]
        if not (
            crosses_dateline(p1[0], p2[0])
            or crosses_dateline(p2[0], p3[0])
            or crosses_dateline(p3[0], p1[0])
        ):
            filtered_cells.append(cell)
    return filtered_cells


# Note: global_stereo.shp has been generated using global_tag() function in pyposeidon
# https://github.com/ec-jrc/pyPoseidon/blob/9cfd3bbf5598c810004def83b1f43dc5149addd0/pyposeidon/boundary.py#L452
fname = "tests/global/global_latlon.shp"
fname2 = "tests/global/global_stereo.shp"

EPSG = 4326  # EPSG:4326 or WGS84
bbox = (-180.00, 180.00, -89.00, 90.00)
extent = om.Region(extent=bbox, crs=4326)

min_edge_length = 0.5  # minimum mesh size in domain in meters
max_edge_length = 2  # maximum mesh size in domain in meters
shoreline = om.Shoreline(fname, extent.bbox, min_edge_length)
sdf = om.signed_distance_function(shoreline)
edge_length = om.distance_sizing_function(shoreline, rate=0.11)

# once the size functions have been defined, wed need to mesh inside domain in
# stereographic projections. This is way we use another coastline which is
# already translated in a sterographic projection
shoreline_stereo = om.Shoreline(fname2, extent.bbox, min_edge_length, stereo=True)
domain = om.signed_distance_function(shoreline_stereo)

points, cells = om.generate_mesh(domain, edge_length, stereo=True, max_iter=100)

# remove degenerate mesh faces and other common problems in the mesh
points, cells = om.make_mesh_boundaries_traversable(points, cells)
points, cells = om.delete_faces_connected_to_one_face(points, cells)

# apply a Laplacian smoother
points, cells = om.laplacian2(points, cells, max_iter=100)
lon, lat = to_lat_lon(points[:, 0], points[:, 1])
trin = filter_triangles(np.array([lon, lat]).T, cells)

fig, ax, pc = edge_length.plot(
    holding=True, plot_colorbar=True, cbarlabel="Resolution in °", cmap="magma"
)
ax.triplot(lon, lat, trin, color="w", linewidth=0.25)
plt.tight_layout()
plt.show()
```

![Global](https://github.com/tomsail/oceanmesh/assets/18373442/a9c45416-78d5-4f3b-a0a3-1afd061d8dbd)

See the tests inside the `testing/` folder for more inspiration. Work is ongoing on this package.

# Testing

To run the `oceanmesh` unit tests (and turn off plots), check out this repository and type `tox`. `tox` can be installed via pip.

# License

This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html)
