@echo off
setlocal

set ENV=OM
if "%~1" NEQ "" set ENV=%~1

where conda > NUL
if errorlevel 1 (
  echo This script is designed for a conda distribution.
  echo If you are using another distro with virtualenv/pip, then you 
  echo will need to follow the steps below [e.g. for python 3.9.x]:
  echo 1. obtain wheels from https://www.lfd.uci.edu/~gohlke/pythonlibs:
  echo    GDAL, Fiona, rasterio
  echo 2. Run:
  echo    pip install GDAL-3.4.1-cp39-cp39-win_amd64.whl
  echo    pip install Fiona-1.8.21-cp39-cp39-win_amd64.whl
  echo    pip install rasterio-1.2.10-cp39-cp39-win_amd64.whl
  echo 3. Run:
  echo    pip install pybind11
  echo    python setup.py install
  pause
  exit /b 1
)

echo creating OceanMesh conda environment "%ENV%"
call conda create --name %ENV%
call conda activate %ENV%

echo installing OceanMesh conda packages
call conda install -c conda-forge geopandas rasterio scikit-fmm pybind11 cython

echo Installing OceanMesh package
python setup.py install

echo Done
pause
exit /b 0
