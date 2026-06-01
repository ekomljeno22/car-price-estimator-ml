#!/bin/bash
# Bash setup script for Car Price Estimator ML project
# Usage: chmod +x setup.sh && ./setup.sh

echo ""
echo "========================================"
echo "Car Price Estimator ML - Setup Script"
echo "========================================"
echo ""

echo "[Step 1] Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    echo "Make sure Python 3.8+ is installed"
    exit 1
fi

echo "[Step 2] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "[Step 3] Upgrading pip..."
python -m pip install --upgrade pip > /dev/null 2>&1

echo "[Step 4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Virtual environment created successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, simply type:"
echo "  deactivate"
echo ""
echo "Next steps:"
echo "  1. Download used_cars.csv from Kaggle"
echo "  2. Place it in the project root directory"
echo "  3. Run: python 01_eda.py"
echo ""
