"""
Run this ONCE after training to persist the target-encoding maps.
Place this file next to your training scripts and run:

    python export_te_maps.py

It reads used_cars.csv, re-computes the TE maps on the training split,
and saves  data/te_maps.pkl  used by the FastAPI service.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)

df = pd.read_csv("used_cars.csv")

df["price"]  = pd.to_numeric(df["price"].str.replace("[$,]", "", regex=True), errors="coerce")
df["milage"] = pd.to_numeric(df["milage"].str.replace(r"\s*mi\.", "", regex=True).str.replace(",", ""), errors="coerce")
df = df.dropna(subset=["price", "milage", "model_year"])

y = np.log1p(df["price"])
X = df.copy()

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

global_mean = float(y_train.mean())
te_maps = {"global_mean": global_mean}

for col in ["brand", "model"]:
    tmp = X_train.copy()
    tmp["_target"] = y_train
    te_maps[col] = tmp.groupby(col)["_target"].mean()

with open("data/te_maps.pkl", "wb") as f:
    pickle.dump(te_maps, f)

print(f"Saved te_maps.pkl — brands: {len(te_maps['brand'])}, models: {len(te_maps['model'])}")