@echo off
setlocal
echo ========================================
echo Fusion-Based Accident Predictor
echo ========================================
echo.
echo Starting application...
echo Browser will open automatically.
echo.
echo Press Ctrl+C to stop the application.
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

%PY_EXE% -m streamlit run app.py
pause
