@echo off
setlocal
echo ========================================
echo Fusion-Based Accident Predictor Setup
echo ========================================
echo.

set "PY_EXE="

:: 1. Try py launcher
py -c "import sys" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=py"
    goto :found_python
)

:: 2. Try python command in PATH
python -c "import sys" >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=python"
    goto :found_python
)

:: 3. Try standard user Python installs
for %%V in (313 312 311 310) do (
    if exist "%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe" (
        set "PY_EXE=%LOCALAPPDATA%\Programs\Python\Python%%V\python.exe"
        goto :found_python
    )
)

:: 4. Try standard system Python installs
for %%V in (313 312 311 310) do (
    if exist "C:\Python%%V\python.exe" (
        set "PY_EXE=C:\Python%%V\python.exe"
        goto :found_python
    )
)

:found_python
if "%PY_EXE%"=="" (
    echo [ERROR] Python not found on this machine!
    echo Please install Python 3.10+ from python.org and check 'Add to PATH'.
    pause
    exit /b 1
)

echo Using Python: %PY_EXE%
echo Installing Python dependencies...
%PY_EXE% -m pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application, run RUN_APP.bat
echo.
pause
