"""
FastAPI ML microservice – wraps the trained sklearn model.
Exposes POST /predict endpoint consumed by the ASP.NET Core backend.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI(title="Used Cars ML Service", version="1.0.0")

# ── Load artefacts at startup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
TE_MAP_PATH    = os.path.join(BASE_DIR, "..", "data",   "te_maps.pkl")
RARE_MAPS_PATH = os.path.join(BASE_DIR, "..", "data",   "rare_maps.pkl")

# MLP Enkoderi (Pipeline 2)
SCALER_PATH    = os.path.join(BASE_DIR, "..", "data",   "scaler.pkl")
OHE_PATH       = os.path.join(BASE_DIR, "..", "data",   "ohe.pkl")

# HGB Enkoder (Pipeline 1)
ORD_ENC_PATH   = os.path.join(BASE_DIR, "..", "data",   "ord_enc.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        MODEL_DICT = pickle.load(f)
        MODEL = MODEL_DICT['model']          # <-- POPRAVLJENO: Izvlačenje iz dict-a
        MODEL_TYPE = MODEL_DICT['type']      # 'HistGradientBoosting' ili 'MLP'
        
    with open(TE_MAP_PATH, "rb") as f:
        TE_MAPS = pickle.load(f)
        
    with open(RARE_MAPS_PATH, "rb") as f:
        RARE_MAPS = pickle.load(f)
        
    with open(SCALER_PATH, "rb") as f:
        SCALER = pickle.load(f)
        
    with open(OHE_PATH, "rb") as f:
        OHE = pickle.load(f)
        
    with open(ORD_ENC_PATH, "rb") as f:
        ORD_ENC = pickle.load(f)

    print(f"✅ Model ({MODEL_TYPE}) & preprocessors loaded successfully")
except FileNotFoundError as e:
    print(f"⚠️ Could not load artefact: {e}")
    MODEL = MODEL_TYPE = TE_MAPS = RARE_MAPS = SCALER = OHE = ORD_ENC = None


# ── Request / response schemas ───────────────────────────────────────────────
class PredictRequest(BaseModel):
    model_year:   int
    milage:       float
    fuel_type:    str = "gas"
    transmission: str = "automatic"
    accident:     str = "None"
    clean_title:  str = "Yes"
    brand:        str = "Toyota"
    model:        str = "Camry"


class PredictResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    model_used: str


# ── Helper ───────────────────────────────────────────────────────────────────
def apply_target_encoding(value: str, te_series, global_mean: float) -> float:
    """Map a raw string to its target-encoded float."""
    try:
        return float(te_series.loc[value])
    except KeyError:
        return global_mean


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None, "model_type": MODEL_TYPE}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or TE_MAPS is None:
        raise HTTPException(status_code=503, detail="Model elements not loaded.")

    global_mean = TE_MAPS["global_mean"]

    # 1. Repliciranje sirovog Feature Engineeringa (Identično kao u 02_preprocessing)
    CURRENT_YEAR = 2025
    model_age = CURRENT_YEAR - req.model_year
    milage_log = np.log1p(req.milage)
    
    # Sigurno dijeljenje (clip lower=1 kao u treningu)
    denom = max(1, model_age)
    miles_per_year = req.milage / denom

    # 2. Obrada rijetkih kategorija ('Other' zamjena)
    fuel_type = "Other" if req.fuel_type in RARE_MAPS.get('fuel_type', set()) else req.fuel_type
    transmission = "Other" if req.transmission in RARE_MAPS.get('transmission', set()) else req.transmission
    
    # 3. Target Encoding za High-Cardinality značajke
    brand_te = apply_target_encoding(req.brand, TE_MAPS["brand"], global_mean)
    model_te = apply_target_encoding(req.model, TE_MAPS["model"], global_mean)

    # 4. Kreiranje baznih struktura za transformaciju
    # Točan redoslijed num_and_te stupaca: ['model_year', 'model_age', 'milage_log', 'miles_per_year', 'brand', 'model']
    num_and_te_vals = np.array([[req.model_year, model_age, milage_log, miles_per_year, brand_te, model_te]])
    
    # Točan redoslijed kategoričkih stupaca: ['fuel_type', 'transmission', 'accident', 'clean_title']
    cat_vals = [[fuel_type, transmission, req.accident, req.clean_title]]
    cat_df = pd.DataFrame(cat_vals, columns=['fuel_type', 'transmission', 'accident', 'clean_title'])

    # 5. Transformacija ovisno o tome koji je model pobijedio na treningu
    if MODEL_TYPE == 'HistGradientBoosting':
        # Pipeline 1: OrdinalEncoder za kategorije + spajanje s numeričkima bez skaliranja
        cat_enc = ORD_ENC.transform(cat_df).astype(int)
        X_final = np.hstack([num_and_te_vals, cat_enc])
    else:
        # Pipeline 2 (MLP): StandardScaler za numeričke + OneHotEncoder za kategorije
        num_scaled = SCALER.transform(num_and_te_vals)
        cat_ohe = OHE.transform(cat_df)
        X_final = np.hstack([num_scaled, cat_ohe])

    # 6. Predikcija modela i inverzija logaritma
    y_log = MODEL.predict(X_final)[0]
    price = float(np.expm1(y_log))
    
    # Spriječi negativne cijene u ekstremnim anomalijama ulaza
    price = max(2500.0, price) 

    return PredictResponse(
        predicted_price=round(price, 2),
        predicted_price_formatted=f"${price:,.2f}",
        model_used=MODEL_TYPE
    )


@app.get("/options")
def options():
    """Return valid dropdown values for the UI."""
    brands = list(TE_MAPS["brand"].index) if TE_MAPS else []
    models = list(TE_MAPS["model"].index) if TE_MAPS else []
    return {
        "fuel_types":    ["gas", "diesel", "electric", "hybrid", "Other"],
        "transmissions": ["automatic", "manual", "Other"],
        "accidents":     ["None", "Yes"],
        "clean_titles":  ["Yes", "No"],
        "brands":        sorted(brands),
        "models":        sorted(models),
    }

@app.get("/models-for-brand/{brand}")
def models_for_brand(brand: str):
    """Vraća samo modele koji postoje za zadani brand."""
    if TE_MAPS is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
 
    brand_model_map: dict = TE_MAPS.get("brand_model_map", {})
 
    # Tražimo brand case-insensitive
    models = brand_model_map.get(brand)
    if models is None:
        # Pokušaj case-insensitive lookup
        for key in brand_model_map:
            if key.lower() == brand.lower():
                models = brand_model_map[key]
                break
 
    if models is None:
        raise HTTPException(
            status_code=404,
            detail=f"Brand '{brand}' not found. Run export_te_maps.py to rebuild maps."
        )
 
    return {"brand": brand, "models": sorted(models)}