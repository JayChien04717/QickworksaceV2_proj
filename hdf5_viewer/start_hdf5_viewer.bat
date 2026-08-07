@echo off
setlocal
set "VIEWER_PYTHON=C:\Users\cluster\anaconda3\envs\qick_gui\python.exe"
set "VIEWER_URL=http://127.0.0.1:8765"

if not exist "%VIEWER_PYTHON%" (
  set "VIEWER_PYTHON=python"
)

"%VIEWER_PYTHON%" -c "import fastapi, h5py, numpy, uvicorn, watchfiles" >nul 2>&1
if errorlevel 1 (
  echo Viewer dependencies are missing from: %VIEWER_PYTHON%
  echo Required: fastapi h5py numpy uvicorn watchfiles
  pause
  exit /b 1
)

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%VIEWER_URL%/' -TimeoutSec 2; if ($r.Content -match 'Quantum Data Library') { exit 0 }; exit 2 } catch { exit 1 }" >nul 2>&1
set "VIEWER_STATUS=%ERRORLEVEL%"
if "%VIEWER_STATUS%"=="0" (
  echo HDF5 viewer is already running: %VIEWER_URL%
  if not defined VIEWER_NO_BROWSER start "" "%VIEWER_URL%"
  exit /b 0
)
if "%VIEWER_STATUS%"=="2" (
  echo Port 8765 is occupied by another application.
  pause
  exit /b 1
)

pushd "%~dp0"
echo HDF5 viewer: %VIEWER_URL%
echo Auto-reload is enabled for Python and HTML changes.
echo Press Ctrl+C to stop the viewer.
if not defined VIEWER_NO_BROWSER start "" "%VIEWER_URL%"
"%VIEWER_PYTHON%" -m uvicorn hdf5_viewer_server:app --host 127.0.0.1 --port 8765 --reload --reload-dir "." --reload-include "*.py" --reload-include "*.html"
if errorlevel 1 (
  echo Viewer failed to start. See the error above.
  pause
)
popd