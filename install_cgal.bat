@echo off
setlocal

REM This is because the latest vcpkg uses a portable python 3.10 that wont run on Windows 7
REM
REM CMAKE is required here only because vcpkg uses it to build CGAL and its dependencies.
REM OceanMesh itself does NOT use CMAKE for its Python build; it relies on setuptools + pybind11.
REM See setup.py for details of the extension build process.
for /F %%v in ('powershell -Command [environment]::OSVersion.Version.Major') do set OS_VER=%%v
if "%OS_VER%" LSS "10" (
  echo This install script requires Windows 10 or later [you have version "%OS_VER%"]
  echo You will need to install CGAL by other means.
  goto FAIL
)

REM This was for Windows 7... Windows 10 comes with a suitable powershell version
REM
for /F %%v in ('powershell -Command $Host.Version.Major') do set PS_VER=%%v
if "%PS_VER%" LSS "3" (
  echo This install script requires Windows Powershell Version 3 or later [you have version "%PS_VER%"]
  echo See "https://docs.microsoft.com/en-us/powershell/"
  goto FAIL
)

REM check prerequisites to build C/C++ using vcpkg
REM
if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2015" (
  if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2017" (
    if not exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2019" (
      if not exist "%ProgramFiles%\Microsoft Visual Studio\2022" (
        echo This install script requires "Visual Studio"
        echo See "https://visualstudio.microsoft.com/downloads/"
        goto FAIL
      )
    )
  )
)
if not exist "%ProgramFiles%\CMake" (
  echo This install script requires "CMake"
  echo See "https://cmake.org/download/"
  goto FAIL
)
if not exist "%ProgramFiles%\Git" (
  echo This install script requires "git"
  echo See "https://git-scm.com/download/"
  goto FAIL
)

REM get cgal libraries from vcpkg and build
REM
set INSTALL_DIR=%USERPROFILE%\OceanMesh
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

pushd "%INSTALL_DIR%"

  set LOG=%INSTALL_DIR%\build.log
  
  set FRESH=NO
  if not exist vcpkg set FRESH=YES

  if "%FRESH%" EQU "YES" git clone https://github.com/Microsoft/vcpkg.git

  if not exist vcpkg (
    echo 'git clone' failed.. if you have just installed git, then close this command prompt before re-running this script.
    popd
    goto FAIL
  )
  
  pushd vcpkg
    if "%FRESH%" EQU "YES" (
      call bootstrap-vcpkg.bat
    ) else (
      vcpkg upgrade --no-dry-run
    )
    echo building cgal... this generally takes several minutes.
    vcpkg install cgal:x64-windows > "%LOG%"
  popd
  
popd

set BIN_DIR=%INSTALL_DIR%\vcpkg\installed\x64-windows\bin

if not exist "%BIN_DIR%\gmp.dll" (
  echo build has failed or is incomplete... see "%LOG%".
  goto FAIL
)

REM permanently set environment variable to location of DLLs
REM
echo build seems to have succeeded.
endlocal & set CGAL_BIN=%BIN_DIR%
setx CGAL_BIN "%CGAL_BIN%"
pause
exit /b 0

:FAIL
pause
exit /b 1
