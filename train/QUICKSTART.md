# QUICK START GUIDE

## 5-Minute Setup

### Step 1: Activate Setup (Choose One)

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Windows (Command Prompt):**
```cmd
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Step 2: Download Dataset

1. Go to: https://www.kaggle.com/datasets/taeefnajib/used-car-price-prediction-dataset
2. Click "Download" 
3. Extract and place `used_cars.csv` in project root

### Step 3: Run the Pipeline

```bash
# Make sure venv is activated, then:
python 01_eda.py
python 02_preprocessing.py
python 03_build_model.py
python 04_train_model.py
python 05_evaluate_model.py
python 06_visualize_curves.py
python 07_feature_importance.py
```