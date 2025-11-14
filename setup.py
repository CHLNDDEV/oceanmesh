import os
import sys
import configparser
from setuptools import Extension

# Try to import Cython for optional inpoly speedup
try:
    from Cython.Build import cythonize

    CYTHON_AVAILABLE = True
except Exception:
    CYTHON_AVAILABLE = False

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup  # , find_packages

import versioneer

sys.path.append(os.path.dirname(__file__))

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

# Optional Cython extension for vendored inpoly (point-in-polygon)
# Provides significant speedup over the pure-Python fallback. Build is optional.


def maybe_add_inpoly_extension():
    try:
        import numpy as np
    except Exception:
        # numpy will be available during isolated builds via pyproject.toml
        # but if not, skip adding the optional extension
        return

    inpoly_ext_name = "oceanmesh._vendor.inpoly.inpoly_"
    pyx_path = os.path.join("oceanmesh", "_vendor", "inpoly", "inpoly_.pyx")
    c_path = os.path.join("oceanmesh", "_vendor", "inpoly", "inpoly_.c")

    if CYTHON_AVAILABLE and os.path.exists(pyx_path):
        try:
            inpoly_ext = Extension(
                inpoly_ext_name,
                sources=[pyx_path],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                optional=True,
            )
            inpoly_modules = cythonize(
                [inpoly_ext],
                compiler_directives={
                    "language_level": 3,
                    "boundscheck": False,
                    "wraparound": False,
                    "cdivision": True,
                    "nonecheck": False,
                },
                annotate=False,
            )
            ext_modules.extend(inpoly_modules)
        except Exception as ex:
            print(
                f"[setup.py] Cythonization of {pyx_path} failed: {ex}. "
                "Falling back to pre-generated C source if available."
            )
            if os.path.exists(c_path):
                inpoly_ext = Extension(
                    inpoly_ext_name,
                    sources=[c_path],
                    include_dirs=[np.get_include()],
                    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                    optional=True,
                )
                ext_modules.append(inpoly_ext)
    elif os.path.exists(c_path):
        inpoly_ext = Extension(
            inpoly_ext_name,
            sources=[c_path],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            optional=True,
        )
        ext_modules.append(inpoly_ext)


maybe_add_inpoly_extension()

cmdclass = versioneer.get_cmdclass()
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


if __name__ == "__main__":
    setup(
        install_requires=get_requirements(),
        cmdclass=cmdclass,
        version=versioneer.get_version(),
        ext_modules=ext_modules,
        zip_safe=False,
    )
