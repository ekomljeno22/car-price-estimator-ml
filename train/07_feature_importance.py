import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pickle
import os

os.makedirs('results', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
X_test_proc = np.load('data/X_test_proc.npy')
y_test      = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 60)
print("FEATURE IMPORTANCE ANCESTRY INFO")
print("=" * 60)
print(f"Model architecture : {type(model).__name__}")
print(f"Test matrix shape  : {X_test_proc.shape}")
print("=" * 60)

# ── Extract Feature Names ─────────────────────────────────────────────────────
try:
    with open('data/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    feature_names = preprocessor.get_feature_names_out().tolist()
    print("Loaded feature names from data/preprocessor.pkl")
except FileNotFoundError:
    print("data/preprocessor.pkl not found – reconstructing names from pipeline defaults…")
    
    # Restored 'model_age' to properly match 02_preprocessing.py
    numeric_features     = ['model_year', 'model_age', 'milage_log']
    categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
    high_card_features   = ['brand', 'model']
    num_and_te_features  = numeric_features + high_card_features
    
    feature_names = []
    for feat in num_and_te_features:
        feature_names.append(f'num_and_te__{feat}')
    
    cat_mapping = {
        'fuel_type':    ['diesel', 'electric', 'gas', 'hybrid'],
        'transmission': ['automatic', 'manual'],
        'accident':     ['None', 'Yes'],
        'clean_title':  ['No', 'Yes']
    }
    
    for cat_feat in categorical_features:
        values = cat_mapping.get(cat_feat, [])
        for val in values:
            feature_names.append(f'cat__{cat_feat}_{val}')
    
    print(f"Reconstructed {len(feature_names)} feature names mapping baseline.")

# Shape validation guardrail
if len(feature_names) != X_test_proc.shape[1]:
    print(f"\nWARNING: Expected {X_test_proc.shape[1]} features, but compiled {len(feature_names)}.")
    print("Falling back to generic feature naming conventions.")
    feature_names = [f'feature_{i}' for i in range(X_test_proc.shape[1])]

# ── Permutation Importance ────────────────────────────────────────────────────
print("\nComputing permutation feature importance… (may take 1–3 min)")

importance_results = permutation_importance(
    model,
    X_test_proc,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2',
    n_jobs=-1
)

importances_series = pd.Series(
    importance_results.importances_mean, index=feature_names
).sort_values(ascending=False)

print("\n" + "=" * 60)
print("TOP 15 MOST IMPORTANT FEATURES")
print("=" * 60)
for i, (feature, importance) in enumerate(importances_series.head(15).items(), 1):
    clean_name = feature.split('__')[-1]
    print(f" {i:2d}. {clean_name:<35} | Importance (ΔR²): {importance:.5f}")
print("=" * 60)

# ── Plot & Persist ────────────────────────────────────────────────────────────
plt.figure(figsize=(12, 6))
top_features = importances_series.head(15).sort_values()

# Clean up structural names for clean display labels
plot_labels = [label.split('__')[-1] for label in top_features.index]

colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(top_features)))
plt.barh(plot_labels, top_features.values, color=colors, edgecolor='black', linewidth=0.6)

plt.xlabel('Mean R² decrease after permutation')
plt.ylabel('Input Features')
plt.title('Top 15 Most Important Features (Permutation Importance)')
plt.grid(True, linestyle=':', alpha=0.5, axis='x')
plt.tight_layout()

plt.savefig('results/feature_importance.png', dpi=150)
plt.close()
print("\nSaved: results/feature_importance.png")

importance_df = pd.DataFrame({
    'feature':         importances_series.index,
    'importance_mean': importances_series.values,
    'importance_std':  importance_results.importances_std
}).sort_values('importance_mean', ascending=False)

importance_df.to_csv('results/feature_importance.csv', index=False)
print("Saved: results/feature_importance.csv")