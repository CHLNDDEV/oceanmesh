##############################################################################
# LEGACY/UNSUPPORTED BUILD SCRIPT
#
# This CMake configuration is retained for historical reference only.
# It is NOT used by the standard Python build, which uses setuptools with
# pybind11 (see setup.py and pyproject.toml). Do not rely on this file for
# normal development or installation. The supported build path is:
#   - pip/setuptools (Pybind11Extension) invoked via setup.py
# On Windows, CMake may still be required by vcpkg solely to build CGAL
# dependencies; it is not used to build oceanmesh itself.
##############################################################################

cmake_minimum_required(VERSION 2.8.12)

project(oceanmesh)

##############################################################################

# Add custom CMake modules
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Some screen output
message(STATUS "OS detected: ${CMAKE_SYSTEM_NAME}")
message(STATUS "CXX Compiler detected: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMake additional search path for libraries: ${CMAKE_LIBRARY_PATH}")

# CGAL and its components
find_package(CGAL)
message(STATUS "CGAL version: ${CGAL_VERSION}")

if( CGAL_VERSION  VERSION_LESS 5.0)
    message(FATAL_ERROR "This project requires at least CGAL 5.0 library and will not be compiled.")
endif()


# include helper file
include( ${CGAL_USE_FILE})

set (SRC "oceanmesh/cpp")

include_directories (${SRC})

find_package(pybind11)
message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")

#add_subdirectory(pybind11)
pybind11_add_module(delaunay_class ${SOURCES} "${SRC}/delaunay_class.cpp")
pybind11_add_module(HamiltonJacobi ${SOURCES} "${SRC}/HamiltonJacobi.cpp")
pybind11_add_module(fast_geometry ${SOURCES} "${SRC}/fast_geometry.cpp")
