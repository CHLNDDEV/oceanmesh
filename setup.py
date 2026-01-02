import os
import sys
import configparser

try:
    import numpy as np

    _HAVE_NUMPY = True
except ModuleNotFoundError:  # pragma: no cover
    np = None
    _HAVE_NUMPY = False
from setuptools import Extension, setup  # , find_packages

import versioneer

try:
    from pybind11.setup_helpers import (  # pyright: ignore[reportMissingImports]
        Pybind11Extension,
        build_ext as pybind_build_ext,
    )

    _HAVE_PYBIND11 = True
except ModuleNotFoundError:  # pragma: no cover
    Pybind11Extension = None
    pybind_build_ext = None
    _HAVE_PYBIND11 = False

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


ext_modules = []


def _get_prefix_paths(prefix: str):
    if not prefix:
        return None, None

    include_dir = os.path.join(prefix, "include")
    lib_dir = os.path.join(prefix, "lib")

    have_include = os.path.isdir(include_dir)
    have_lib = os.path.isdir(lib_dir)

    return (include_dir if have_include else None), (lib_dir if have_lib else None)

if _HAVE_PYBIND11:
    if os.name == "nt":
        # Prefer an explicit prefix when building wheels (e.g., via micromamba/conda).
        # This avoids hard-coding a developer-specific vcpkg location.
        prefix = os.environ.get("OCEANMESH_PREFIX") or os.environ.get("CONDA_PREFIX")
        include_dir, lib_dir = _get_prefix_paths(prefix)

        if include_dir and lib_dir:
            ext_modules = [
                Pybind11Extension(
                    loc,
                    [fi],
                    include_dirs=[include_dir],
                    library_dirs=[lib_dir],
                    extra_link_args=[f"/LIBPATH:{lib_dir}"],
                    libraries=["gmp", "mpfr"],
                )
                for fi, loc in zip(files, is_called)
            ]
        else:
            home = os.environ.get("USERPROFILE", "").replace("\\", "/")
            vcpkg_root = os.environ.get("VCPKG_ROOT") or os.environ.get(
                "VCPKG_INSTALLATION_ROOT"
            )
            if vcpkg_root:
                vcpkg_root = vcpkg_root.replace("\\", "/")
                vcpkg = f"{vcpkg_root}/installed/x64-windows"
            else:
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
        # Ensure compiler/linker can find dependency headers and libs when building
        # wheels (CI) without requiring system-wide installs.
        prefix = os.environ.get("OCEANMESH_PREFIX") or os.environ.get("CONDA_PREFIX")
        include_dir, lib_dir = _get_prefix_paths(prefix)

        include_dirs = []
        library_dirs = []
        extra_link_args = []

        if include_dir:
            include_dirs.append(include_dir)

        if lib_dir:
            library_dirs.append(lib_dir)
            extra_link_args.extend([f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}"])

        ext_modules = [
            Pybind11Extension(
                loc,
                [fi],
                include_dirs=include_dirs,
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

    if not _HAVE_NUMPY:
        return []

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

if _HAVE_PYBIND11:
    cmdclass.update({"build_ext": pybind_build_ext})
else:
    from setuptools.command.build_ext import build_ext as _setuptools_build_ext

    class build_ext(_setuptools_build_ext):
        def run(self):
            raise ModuleNotFoundError(
                "pybind11 is required to build oceanmesh C++ extensions. "
                "Install it (e.g. `python -m pip install pybind11`) or build via PEP 517 "
                "so build dependencies are installed automatically."
            )

    cmdclass.update({"build_ext": build_ext})


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
