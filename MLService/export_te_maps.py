"""
export_te_maps.py – Pokreni JEDANPUT nakon treniranja.
Sprema TE mape i brand→modeli mapiranje u data/te_maps.pkl.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

df = pd.read_csv("used_cars.csv")

df["price"]  = pd.to_numeric(df["price"].str.replace(r"[$,]", "", regex=True), errors="coerce")
df["milage"] = pd.to_numeric(df["milage"].str.replace(r"[^\d.]", "", regex=True), errors="coerce")
df = df.dropna(subset=["price", "milage", "model_year"])
df = df[df["price"] >= 2500].reset_index(drop=True)

y = np.log1p(df["price"])
X = df.copy()

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

global_mean = float(y_train.mean())
te_maps = {"global_mean": global_mean}

for col in ["brand", "model"]:
    tmp = X_train.copy()
    tmp["_target"] = y_train
    te_maps[col] = tmp.groupby(col)["_target"].mean()

# ── NOVO: brand → lista modela iz cijelog dataseta ────────────────────────────
brand_model_map: dict[str, list[str]] = (
    df.groupby("brand")["model"]
      .apply(lambda s: sorted(s.unique().tolist()))
      .to_dict()
)
te_maps["brand_model_map"] = brand_model_map

with open("data/te_maps.pkl", "wb") as f:
    pickle.dump(te_maps, f)

total_models = sum(len(v) for v in brand_model_map.values())
print(f"Saved te_maps.pkl")
print(f"  Brands : {len(te_maps['brand'])}")
print(f"  Models : {len(te_maps['model'])}")
print(f"  Brand→model mapping: {len(brand_model_map)} brandova, {total_models} ukupno modela")