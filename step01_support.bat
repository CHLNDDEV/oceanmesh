@echo off

REM get cgal libraries from vcpkg without ANY other cruft
REM requires: vs2019, git, cmake

if exist support rd /S /Q support
md support
pushd support
  git clone https://github.com/Microsoft/vcpkg.git
  pushd vcpkg
    call bootstrap-vcpkg.bat
    vcpkg install cgal:x64-windows
  popd
popd

pause
