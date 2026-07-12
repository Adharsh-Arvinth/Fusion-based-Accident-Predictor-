@echo off
echo ========================================
echo Fusion-Based Accident Predictor Setup
echo ========================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application, run:
echo   python -m streamlit run app.py
echo.
echo Don't forget to:
echo 1. Update your Gemini API key in app.py
echo 2. Allow camera access when prompted
echo.
pause
