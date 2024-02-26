@echo off
setlocal

REM run package command for vcpkg
REM
if "%~2" EQU "" goto USAGE

set OPT=--overlay-ports="%CD%\ports"
set VCP=%CD%\vcpkg

if "%1" EQU "remove" (
  pushd "%VCP%"
  vcpkg remove %~2:x64-windows %OPT% --recurse
  popd
) else if "%1" EQU "install" (
  pushd "%VCP%"
  vcpkg install %~2:x64-windows %OPT% --binarysource=clear
  popd
  if not exist "%VCP%\installed\x64-windows\share\%~2" (
    echo package install command likely failed for %~2
    GOTO FAIL
  )
) else (
  goto USAGE
)


:SUCCESS
exit /b 0

:USAGE
echo Usage: package install^|remove packagename

:FAIL
exit /b 1
