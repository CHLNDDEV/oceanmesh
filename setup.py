import os
import sys
import configparser

import numpy as np
from pybind11.setup_helpers import Pybind11Extension, build_ext as pybind_build_ext
from setuptools import Extension, setup  # , find_packages

import versioneer

sys.path.append(os.path.dirname(__file__))

# Build system: This package uses pybind11's setuptools integration to compile
# C++ extensions (HamiltonJacobi, delaunay_class, fast_geometry). CMake is NOT
# invoked for building oceanmesh itself; on Windows it may be required only by
# external tools (e.g., vcpkg) to build CGAL dependencies.

# https://github.com/pybind/python_example/
is_called = [
    "_HamiltonJacobi",
    "_delaunay_class",
    "_fast_geometry",
]

files = [
    "oceanmesh/cpp/HamiltonJacobi.cpp",
    "oceanmesh/cpp/delaunay_class.cpp",
    "oceanmesh/cpp/fast_geometry.cpp",
]

if os.name == "nt":
    home = os.environ["USERPROFILE"].replace("\\", "/")
    vcpkg = f"{home}/OceanMesh/vcpkg/installed/x64-windows"
    ext_modules = [
        Pybind11Extension(
            loc,
            [fi],
            include_dirs=[f"{vcpkg}/include"],
            extra_link_args=[f"/LIBPATH:{vcpkg}/lib"],
            libraries=["gmp", "mpfr"],
        )
        for fi, loc in zip(files, is_called)
    ]
else:
    # no CGAL libraries necessary from CGAL 5.0 onwards
    # Ensure linker can find conda-provided lib paths on macOS/Linux
    conda_prefix = os.environ.get("CONDA_PREFIX", sys.prefix)
    conda_lib = os.path.join(conda_prefix, "lib")
    have_libdir = os.path.isdir(conda_lib)

    extra_link_args = []
    library_dirs = []
    if have_libdir:
        library_dirs = [conda_lib]
        # Help the linker locate shared libs at build and at runtime
        extra_link_args = [f"-L{conda_lib}", f"-Wl,-rpath,{conda_lib}"]

    ext_modules = [
        Pybind11Extension(
            loc,
            [fi],
            libraries=["gmp", "mpfr"],
            library_dirs=library_dirs,
            extra_link_args=extra_link_args,
        )
        for fi, loc in zip(files, is_called)
    ]


def maybe_add_cython_extensions():
    """Optionally build Cython extension for point-in-polygon acceleration.

    Uses the presence of either a .pyx (when building from a git checkout
    with Cython available) or a pre-generated .c file (when building from
    an sdist) to decide whether to add the extension. If neither is present
    the build proceeds without the accelerated kernel.
    """

    from pathlib import Path

    src_dir = Path(__file__).parent / "oceanmesh" / "geometry"
    pyx_path = src_dir / "point_in_polygon_.pyx"
    c_path = src_dir / "point_in_polygon_.c"

    sources = None
    if pyx_path.is_file():
        sources = [str(pyx_path)]
    elif c_path.is_file():
        sources = [str(c_path)]
    else:
        return []

    extra_compile_args = []
    if sys.platform == "win32":
        extra_compile_args = ["/O2"]
    else:
        extra_compile_args = ["-O3"]

    ext = Extension(
        "oceanmesh.geometry.point_in_polygon_",
        sources=sources,
        include_dirs=[np.get_include()],
        extra_compile_args=extra_compile_args,
    )

    return [ext]


cmdclass = versioneer.get_cmdclass()
cmdclass.update({"build_ext": pybind_build_ext})


def get_requirements():
    """
    Fix
    """

    config = configparser.ConfigParser()
    config.read("setup.cfg")
    requirements = config["options"]["install_requires"].split()

    if sys.version_info < (3, 9):
        requirements.remove("fiona")
        requirements.append("fiona<1.10")

    return requirements


cython_exts = maybe_add_cython_extensions()
ext_modules.extend(cython_exts)

try:
    from Cython.Build import cythonize

    have_cython = True
except ImportError:  # pragma: no cover - Cython not installed
    have_cython = False

if have_cython and cython_exts:
    cythonized = cythonize(cython_exts, compiler_directives={"language_level": "3"})
    # Replace the plain Cython extensions in ext_modules with the
    # cythonized ones, leaving the pybind11 extensions untouched.
    ext_modules = [e for e in ext_modules if e not in cython_exts] + list(cythonized)


if __name__ == "__main__":
    setup(
        install_requires=get_requirements(),
        cmdclass=cmdclass,
        version=versioneer.get_version(),
        ext_modules=ext_modules,
        zip_safe=False,
    )
