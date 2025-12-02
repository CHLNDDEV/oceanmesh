import os

# DPZ patches for miniconda on windows using vcpkg to install cgal and dependencies
#
if os.name == "nt":
    assert os.environ.get(
        "CGAL_BIN", None
    ), "The environment variable CGAL_BIN must be set."
    os.add_dll_directory(os.environ["CGAL_BIN"])

from oceanmesh.boundary import identify_ocean_boundary_sections
from oceanmesh.clean import (
    delete_boundary_faces,
    delete_exterior_faces,
    delete_faces_connected_to_one_face,
    delete_interior_faces,
    laplacian2,
    make_mesh_boundaries_traversable,
    mesh_clean,
)
from oceanmesh.edgefx import (
    bathymetric_gradient_sizing_function,
    distance_sizing_from_line_function,
    distance_sizing_from_point_function,
    distance_sizing_function,
    enforce_mesh_gradation,
    enforce_mesh_size_bounds_elevation,
    feature_sizing_function,
    multiscale_sizing_function,
    wavelength_sizing_function,
)
from oceanmesh.edges import draw_edges, get_poly_edges
from oceanmesh.filterfx import filt2
from oceanmesh.geodata import (
    DEM,
    Shoreline,
    create_circle_coords,
    get_polygon_coordinates,
)
from oceanmesh.grid import Grid, compute_minimum
from oceanmesh.region import (
    Region,
    stereo_to_3d,
    to_3d,
    to_lat_lon,
    to_stereo,
    warp_coordinates,
)
from oceanmesh.signed_distance_function import (
    Difference,
    Domain,
    Intersection,
    Union,
    create_bbox,
    create_circle,
    multiscale_signed_distance_function,
    signed_distance_function,
)

from .fix_mesh import fix_mesh, simp_vol, simp_qual
from .mesh_generator import (
    generate_mesh,
    generate_multiscale_mesh,
    plot_mesh_bathy,
    plot_mesh_connectivity,
    write_to_fort14,
    write_to_t3s,
)

__all__ = [
    "create_bbox",
    "Region",
    "stereo_to_3d",
    "to_lat_lon",
    "to_3d",
    "to_stereo",
    "compute_minimum",
    "create_circle_coords",
    "bathymetric_gradient_sizing_function",
    "multiscale_sizing_function",
    "delete_boundary_faces",
    "delete_faces_connected_to_one_face",
    "distance_sizing_from_point_function",
    "distance_sizing_from_line_function",
    "plot_mesh_connectivity",
    "plot_mesh_bathy",
    "make_mesh_boundaries_traversable",
    "enforce_mesh_size_bounds_elevation",
    "laplacian2",
    "delete_interior_faces",
    "delete_exterior_faces",
    "mesh_clean",
    "SizeFunction",
    "Grid",
    "DEM",
    "Domain",
    "create_circle",
    "crate_bbox",
    "Union",
    "Difference",
    "Intersection",
    "identify_ocean_boundary_sections",
    "Shoreline",
    "generate_multiscale_mesh",
    "get_polygon_coordinates",
    "distance_sizing_function",
    "feature_sizing_function",
    "enforce_mesh_gradation",
    "wavelength_sizing_function",
    "slope_sizing_function",
    "multiscale_signed_distance_function",
    "signed_distance_function",
    "filt2",
    "get_poly_edges",
    "draw_edges",
    "generate_mesh",
    "fix_mesh",
    "simp_vol",
    "simp_qual",
    "warp_coordinates",
    "write_to_fort14",
    "write_to_t3s",
]

from . import _version

__version__ = _version.get_versions()["version"]

try:  # Optional global-stereo helpers (import may fail generically)
    from .projections import CARTOPY_AVAILABLE

    __all__.extend(["StereoProjection", "CARTOPY_AVAILABLE"])
except ImportError:
    # Projections module not importable; expose CARTOPY_AVAILABLE flag as False
    CARTOPY_AVAILABLE = False
    __all__.append("CARTOPY_AVAILABLE")
