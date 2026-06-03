"""
FastAPI ML microservice – wraps the trained sklearn MLPRegressor.
Exposes POST /predict endpoint consumed by the ASP.NET Core backend.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI(title="Used Cars ML Service", version="1.0.0")

# ── Load artefacts at startup ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
PREP_PATH  = os.path.join(BASE_DIR, "..", "data",   "preprocessor.pkl")
TE_MAP_PATH = os.path.join(BASE_DIR, "..", "data",  "te_maps.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        MODEL = pickle.load(f)
    with open(PREP_PATH, "rb") as f:
        PREPROCESSOR = pickle.load(f)
    with open(TE_MAP_PATH, "rb") as f:
        TE_MAPS = pickle.load(f)   # dict: { "brand": Series, "model": Series, "global_mean": float }
    print("✅  Model & preprocessor loaded successfully")
except FileNotFoundError as e:
    print(f"⚠️  Could not load artefact: {e}")
    MODEL = None
    PREPROCESSOR = None
    TE_MAPS = None


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


# ── Helper ───────────────────────────────────────────────────────────────────
def apply_target_encoding(value: str, te_series, global_mean: float) -> float:
    """Map a raw string to its target-encoded float."""
    return float(te_series.get(value, global_mean))


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None or PREPROCESSOR is None or TE_MAPS is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    import pandas as pd

    global_mean = TE_MAPS["global_mean"]

    milage_log = np.log1p(req.milage)

    brand_enc = apply_target_encoding(req.brand, TE_MAPS["brand"], global_mean)
    model_enc = apply_target_encoding(req.model, TE_MAPS["model"], global_mean)

    # Build a 1-row DataFrame matching the preprocessor's expected input
    row = pd.DataFrame([{
        "model_year":   req.model_year,
        "milage_log":   milage_log,
        "brand":        brand_enc,
        "model":        model_enc,
        "fuel_type":    req.fuel_type,
        "transmission": req.transmission,
        "accident":     req.accident,
        "clean_title":  req.clean_title,
    }])

    X_proc = PREPROCESSOR.transform(row)
    y_log  = MODEL.predict(X_proc)[0]
    price  = float(np.expm1(y_log))

    return PredictResponse(
        predicted_price=round(price, 2),
        predicted_price_formatted=f"${price:,.2f}"
    )


@app.get("/options")
def options():
    """Return valid dropdown values for the UI."""
    brands = list(TE_MAPS["brand"].index) if TE_MAPS else []
    models = list(TE_MAPS["model"].index) if TE_MAPS else []
    return {
        "fuel_types":    ["gas", "diesel", "electric", "hybrid"],
        "transmissions": ["automatic", "manual"],
        "accidents":     ["None", "Yes"],
        "clean_titles":  ["Yes", "No"],
        "brands":        sorted(brands),
        "models":        sorted(models),
    }