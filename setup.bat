@echo off
setlocal
echo ========================================
echo Fusion-Based Accident Predictor Setup
echo ========================================
echo.

set "PY_EXE="
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "PY_EXE=py"
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "PY_EXE=python"
    ) else (
        if exist "C:\Users\adhar\AppData\Local\Programs\Python\Python313\python.exe" (
            set "PY_EXE=C:\Users\adhar\AppData\Local\Programs\Python\Python313\python.exe"
        )
    )
)

if "%PY_EXE%"=="" (
    echo [ERROR] Python not found on this machine!
    echo Please install Python 3.10+ from python.org.
    pause
    exit /b 1
)

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
