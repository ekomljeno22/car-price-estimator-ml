# PowerShell setup script for Car Price Estimator ML project
# Usage: .\setup.ps1

Write-Host ""
Write-Host "========================================"
Write-Host "Car Price Estimator ML - Setup Script"
Write-Host "========================================"
Write-Host ""

Write-Host "[Step 1] Creating virtual environment..."
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create virtual environment"
    Write-Host "Make sure Python 3.8+ is installed and in PATH"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Step 2] Activating virtual environment..."
& .\venv\Scripts\Activate.ps1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to activate virtual environment"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[Step 3] Upgrading pip..."
python -m pip install --upgrade pip | Out-Null

Write-Host "[Step 4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host "Setup Complete!"
Write-Host "========================================"
Write-Host ""
Write-Host "Virtual environment created successfully!"
Write-Host ""
Write-Host "To activate the environment in the future, run:"
Write-Host "  .\venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "To deactivate, simply type:"
Write-Host "  deactivate"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download used_cars.csv from Kaggle"
Write-Host "  2. Place it in the project root directory"
Write-Host "  3. Run: python 01_eda.py"
Write-Host ""
Read-Host "Press Enter to continue"
