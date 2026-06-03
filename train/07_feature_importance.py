import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import pickle, os

os.makedirs('results', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
y_test = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

model      = bundle['model']  if isinstance(bundle, dict) else bundle
model_name = bundle['type']   if isinstance(bundle, dict) else type(bundle).__name__

is_hgb = 'Hist' in model_name or 'Gradient' in type(model).__name__
X_test = np.load('data/X_test_hgb.npy' if is_hgb else 'data/X_test_proc.npy')

with open('data/pipeline_meta.pkl', 'rb') as f:
    meta = pickle.load(f)

feature_names_hgb = meta.get('feature_names_hgb', [f'f_{i}' for i in range(X_test.shape[1])])
feature_names_mlp = meta.get('feature_names_mlp', [f'f_{i}' for i in range(X_test.shape[1])])
feature_names     = feature_names_hgb if is_hgb else feature_names_mlp

print("=" * 65)
print(f"Model: {model_name}")
print(f"Test matrix: {X_test.shape}")
print("=" * 65)

# ── Feature Importance ────────────────────────────────────────────────────────
if is_hgb and hasattr(model, 'feature_importances_'):
    # HistGBM ugrađena važnost (puno brže od permutation)
    importances = model.feature_importances_
    std         = np.zeros(len(importances))
    method      = "Built-in (split gain)"
    print("Koristim ugrađenu feature_importances_ (HistGBM)")
else:
    # MLP: permutation importance, n_jobs=1 (Windows safe)
    print("Računam permutation importance… (1–3 min)")
    result      = permutation_importance(model, X_test, y_test,
                                         n_repeats=10, random_state=42,
                                         scoring='r2', n_jobs=1)
    importances = result.importances_mean
    std         = result.importances_std
    method      = "Permutation (ΔR²)"

# Provjera duljine
if len(feature_names) != len(importances):
    print(f"WARNING: {len(feature_names)} names vs {len(importances)} features → generic names")
    feature_names = [f'feature_{i}' for i in range(len(importances))]

imp_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print(f"\nTOP 15 – {method}")
print("=" * 65)
for i, (feat, imp) in enumerate(imp_series.head(15).items(), 1):
    clean = feat.split('__')[-1]
    print(f"  {i:2d}. {clean:<40} | {imp:+.5f}")
print("=" * 65)

# ── Plot ──────────────────────────────────────────────────────────────────────
top15 = imp_series.head(15).sort_values(ascending=True)
labels = [l.split('__')[-1] for l in top15.index]
colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(top15)))

plt.figure(figsize=(12, 6))
plt.barh(labels, top15.values, color=colors, edgecolor='black', linewidth=0.5)
plt.xlabel(f'Feature Importance ({method})')
plt.title(f'Top 15 Most Important Features – {model_name}')
plt.grid(True, linestyle=':', alpha=0.5, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150)
plt.close()
print("Saved: results/feature_importance.png")

pd.DataFrame({
    'feature':         imp_series.index,
    'importance_mean': imp_series.values,
    'importance_std':  [std[list(feature_names).index(f)]
                        if f in feature_names else 0
                        for f in imp_series.index],
}).to_csv('results/feature_importance.csv', index=False)
print("Saved: results/feature_importance.csv")