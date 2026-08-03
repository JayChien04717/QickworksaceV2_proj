@echo off
setlocal
set "VIEWER_PYTHON=C:\Users\cluster\anaconda3\envs\qick_gui\python.exe"

if not exist "%VIEWER_PYTHON%" (
  echo Python environment not found: %VIEWER_PYTHON%
  pause
  exit /b 1
)

echo HDF5 viewer: http://127.0.0.1:8765
echo Press Ctrl+C to stop the viewer.
"%VIEWER_PYTHON%" "%~dp0hdf5_viewer_server.py"
if errorlevel 1 (
  echo Viewer failed to start. See the error above.
  pause
)
