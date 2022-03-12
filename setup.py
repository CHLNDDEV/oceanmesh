import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

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

import os
if os.name == 'nt':
    home = os.environ['USERPROFILE'].replace('\\', '/')
    vcpkg = f'{home}/OceanMesh/vcpkg/installed/x64-windows'
    ext_modules = [
        Pybind11Extension(loc, [fi], 
            include_dirs=[f'{vcpkg}/include'], 
            extra_link_args=[f'/LIBPATH:{vcpkg}/lib'], 
            libraries=["gmp", "mpfr"]
        )
        for fi, loc in zip(files, is_called)
    ]
else:
    # no CGAL libraries necessary from CGAL 5.0 onwards
    ext_modules = [
        Pybind11Extension(loc, [fi], libraries=["gmp", "mpfr"])
        for fi, loc in zip(files, is_called)
    ]

cmdclass = versioneer.get_cmdclass()
cmdclass.update({"build_ext": build_ext})

if __name__ == "__main__":
    setup(
        cmdclass=cmdclass,
        version=versioneer.get_version(),
        ext_modules=ext_modules,
        zip_safe=False,
    )

