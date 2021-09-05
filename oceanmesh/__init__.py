from oceanmesh.clean import (delete_exterior_faces,
                             delete_faces_connected_to_one_face,
                             delete_interior_faces,
                             make_mesh_boundaries_traversable)
from oceanmesh.edgefx import (distance_sizing_function, enforce_mesh_gradation,
                              enforce_mesh_size_bounds_elevation,
                              multiscale_sizing_function,
                              wavelength_sizing_function)
from oceanmesh.edges import draw_edges, get_poly_edges
from oceanmesh.geodata import DEM, Geodata, Shoreline
from oceanmesh.grid import Grid, compute_minimum
from oceanmesh.signed_distance_function import (
    Difference, Domain, Intersection, Union, create_circle,
    multiscale_signed_distance_function, signed_distance_function)

from .fix_mesh import fix_mesh, simp_vol
from .mesh_generator import generate_mesh, generate_multiscale_mesh, plot_mesh

__all__ = [
    "compute_minimum",
    "multiscale_sizing_function",
    "delete_faces_connected_to_one_face",
    "plot_mesh",
    "make_mesh_boundaries_traversable",
    "enforce_mesh_size_bounds_elevation",
    "delete_interior_faces",
    "delete_exterior_faces",
    "SizeFunction",
    "Grid",
    "Geodata",
    "DEM",
    "Domain",
    "create_circle",
    "Union",
    "Difference",
    "Intersection",
    "Shoreline",
    "generate_multiscale_mesh",
    "distance_sizing_function",
    "enforce_mesh_gradation",
    "wavelength_sizing_function",
    "multiscale_signed_distance_function",
    "signed_distance_function",
    "get_poly_edges",
    "draw_edges",
    "generate_mesh",
    "fix_mesh",
    "simp_vol",
    "simp_qual",
]

from . import _version

__version__ = _version.get_versions()["version"]
