@echo off
REM Setup script for Windows Command Prompt
REM Creates virtual environment and installs dependencies

echo.
echo ========================================
echo Car Price Estimator ML - Setup Script
echo ========================================
echo.

echo [Step 1] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.8+ is installed and in PATH
    pause
    exit /b 1
)

echo [Step 2] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo [Step 3] Upgrading pip...
python -m pip install --upgrade pip

echo [Step 4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment created successfully!
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, simply type:
echo   deactivate
echo.
echo Next steps:
echo   1. Download used_cars.csv from Kaggle
echo   2. Place it in the project root directory
echo   3. Run: python 01_eda.py
echo.
pause
