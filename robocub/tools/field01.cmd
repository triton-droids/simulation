@echo off
set "FIELD01_ROOT=%~dp0..\.."
set "PYTHONUTF8=1"
set "TEMP=%FIELD01_ROOT%\.cache\field01_tmp"
set "TMP=%TEMP%"
set "WARP_CACHE_PATH=%FIELD01_ROOT%\.cache\field01_warp"
if not exist "%TEMP%" mkdir "%TEMP%"
"%FIELD01_ROOT%\.cache\field01_venv\Scripts\python.exe" "%~dp0field01" %*
