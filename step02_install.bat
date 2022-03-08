@echo off

call conda create --name OM
call conda activate OM
call conda install geopandas rasterio scikit-fmm pybind11
python setup.py install
call conda deactivate

pause
