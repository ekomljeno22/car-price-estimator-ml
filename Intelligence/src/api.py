from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import os
import json

app = FastAPI(title="Used Cars ML Microservice", version="1.2.0")

MODEL, MODEL_TYPE, TE_MAPS, RARE_MAPS, SCALER, OHE, ORD_ENC, META = [None] * 8

class PredictRequest(BaseModel):
    model_year: int
    milage: float
    hp: float = 200.0
    liters: float = 2.0
    fuel_type: str = "Gasoline"
    transmission: str = "Automatic"
    accident: str = "None"
    clean_title: str = "Yes"
    brand: str = "Toyota"
    model: str = "Camry"

@app.on_event("startup")
def startup_event():
    global MODEL, MODEL_TYPE, TE_MAPS, RARE_MAPS, SCALER, OHE, ORD_ENC, META
    try:
        with open("models/best_model.pkl", "rb") as f:
            bundle = pickle.load(f)
            MODEL = bundle['model'] if isinstance(bundle, dict) else bundle
            MODEL_TYPE = bundle['type'] if isinstance(bundle, dict) else type(bundle).__name__
        with open("data/te_maps.pkl", "rb") as f: TE_MAPS = pickle.load(f)
        with open("data/rare_maps.pkl", "rb") as f: RARE_MAPS = pickle.load(f)
        with open("data/scaler.pkl", "rb") as f: SCALER = pickle.load(f)
        with open("data/ohe.pkl", "rb") as f: OHE = pickle.load(f)
        with open("data/ord_enc.pkl", "rb") as f: ORD_ENC = pickle.load(f)
        with open("data/pipeline_meta.pkl", "rb") as f: META = pickle.load(f)
        print("🚀 [FastAPI] Svi ML artefakti uspješno podignuti u memoriju servisa.")
        
        if hasattr(MODEL, "feature_names_in_"):
            print(f"📋 [MODEL] Model očekuje sljedećih 12 stupaca: {list(MODEL.feature_names_in_)}")
        else:
            print(f"📋 [MODEL] Model nema spremljena imena stupaca, ali očekuje {getattr(MODEL, 'n_features_in_', 'nepoznato')} stupaca.")

    except Exception as e:
        print(f"❌ [FastAPI] Greška pri podizanju modela: {e}")

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model nije spreman.")
        
    try:
        current_year = META.get("current_year", 2026)
        model_age = current_year - req.model_year
        miles_per_year = req.milage / max(1, model_age)
        
        global_mean = TE_MAPS["global_mean"]
        brand_te = TE_MAPS["brand"].get(req.brand, global_mean)
        model_te = TE_MAPS["model"].get(req.model, global_mean)
        
        num_vals = np.array([[
            req.model_year,
            model_age,
            req.milage,
            miles_per_year,
            req.hp,
            req.liters
        ]])
        
        te_vals = np.array([[brand_te, model_te]])
        
        cat_df = pd.DataFrame(
            [[req.fuel_type, req.transmission, req.accident, req.clean_title]], 
            columns=['fuel_type', 'transmission', 'accident', 'clean_title']
        )
        for col in ['fuel_type', 'transmission']:
            if cat_df.loc[0, col] in RARE_MAPS.get(col, set()):
                cat_df.loc[0, col] = 'Other'
        
        encoded_cats = ORD_ENC.transform(cat_df)
        
        X_final = np.hstack([num_vals, te_vals, encoded_cats])
        
        log_price = MODEL.predict(X_final)[0]
        price = np.expm1(log_price)
        price = max(3000.0, min(120000.0, float(price)))
        print(f"\nPriced price: ${price:,.2f} using model: {MODEL_TYPE}")
        return {
            "predicted_price": round(price, 2),
            "predicted_price_formatted": f"${price:,.0f}",
            "model_used": MODEL_TYPE
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/options")
def get_options():
    if not TE_MAPS:
        raise HTTPException(status_code=503, detail="Mape i podaci nisu učitani.")
    
    encoder = ORD_ENC if ORD_ENC is not None else OHE
    try:
        categories = encoder.categories_
        fuel_types = sorted(list(categories[0]))
        transmissions = sorted(list(categories[1]))
        accidents = sorted(list(categories[2]))
        clean_titles = sorted(list(categories[3]))
    except Exception:
        fuel_types = ["Gasoline", "Diesel", "Electric", "Hybrid", "Other"]
        transmissions = ["Automatic", "Manual", "CVT", "Other"]
        accidents = ["None", "At least one accident or damage reported"]
        clean_titles = ["Yes", "No"]

    brand_map = TE_MAPS.get("brand_model_map", {})
    brands = sorted(list(brand_map.keys()))
    all_models = sorted(list({m for models in brand_map.values() for m in models}))

    return {
        "fuel_types": fuel_types,
        "transmissions": transmissions,
        "accidents": accidents,
        "clean_titles": clean_titles,
        "brands": brands,
        "models": all_models
    }

@app.get("/models-for-brand/{brand}")
def get_models_for_brand(brand: str):
    if not TE_MAPS or "brand_model_map" not in TE_MAPS:
        raise HTTPException(status_code=503, detail="Mape brendova nisu učitane.")
    
    brand_map = TE_MAPS["brand_model_map"]
    models = brand_map.get(brand)
    
    if models is None:
        for k, v in brand_map.items():
            if k.lower() == brand.lower():
                return {"brand": k, "models": v}
        raise HTTPException(status_code=404, detail=f"Brand '{brand}' ne postoji u bazi.")
        
    return {"brand": brand, "models": models}

@app.get("/stats")
def get_stats():
    metrics_path = "results/metrics.json"
    
    if not os.path.exists(metrics_path):
        return {
            "model_type": MODEL_TYPE or "HistGradientBoosting",
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "train_mape": 0.0,
            "train_r2": 0.0,
            "training_samples": 4009,
            "best_model": MODEL_TYPE or "HistGradientBoosting"
        }
        
    try:
        with open(metrics_path, "r") as f:
            data = json.load(f)
            
        return {
            "model_type": data.get("model", MODEL_TYPE or "HistGradientBoosting"),
            "mae": data.get("mae", 0.0),
            "rmse": data.get("rmse", 0.0),
            "r2": data.get("r2", 0.0),
            "mape": data.get("mape", 0.0),
            "train_mape": 0.0,
            "train_r2": 0.0,
            "training_samples": 4009,
            "best_model": data.get("model", MODEL_TYPE or "HistGradientBoosting")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Greška pri čitanju metrika: {e}")