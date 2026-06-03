import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
import os

os.makedirs('results', exist_ok=True)

X_test_proc = np.load('data/X_test_proc.npy')
y_test = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("\n========== MODEL INFO ==========")
print(type(model))
print(model)
print("================================\n")

try:
    with open('data/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    feature_names = preprocessor.get_feature_names_out().tolist()
    print("Loaded feature names from preprocessor.pkl")
except FileNotFoundError:
    print("preprocessor.pkl not found - reconstructing feature names from pipeline...")
    
    numeric_features = ['model_year', 'milage_log']
    categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
    high_card_features = ['brand', 'model']
    num_and_te_features = numeric_features + high_card_features
    
    feature_names = []
    
    for feat in num_and_te_features:
        feature_names.append(f'num_and_te__{feat}')
    
    for cat_feat in categorical_features:
        if cat_feat == 'fuel_type':
            values = ['diesel', 'electric', 'gas', 'hybrid']
        elif cat_feat == 'transmission':
            values = ['automatic', 'manual']
        elif cat_feat == 'accident':
            values = ['None', 'Yes']
        elif cat_feat == 'clean_title':
            values = ['No', 'Yes']
        else:
            values = []
        
        for val in values:
            feature_names.append(f'cat__{cat_feat}_{val}')
    
    print(f"Reconstructed {len(feature_names)} feature names")

print("\nComputing permutation feature importance...")
print("(This may take a few minutes)\n")

print("Before permutation importance")

importance_results = permutation_importance(
    model,
    X_test_proc,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring='r2',
    n_jobs=-1
)

print("After permutation importance")

if len(feature_names) != X_test_proc.shape[1]:
    print(f"Warning: Expected {X_test_proc.shape[1]} features but got {len(feature_names)}")
    print("Using generic feature names instead...")
    feature_names = [f'feature_{i}' for i in range(X_test_proc.shape[1])]

importances_series = pd.Series(
    importance_results.importances_mean, index=feature_names
).sort_values(ascending=False)

print("Top 15 Most Important Features:")
print("=" * 70)
for i, (feature, importance) in enumerate(importances_series.head(15).items(), 1):
    print(f"{i:2d}. {feature:45s} | Importance: {importance:8.5f}")
print("=" * 70)

plt.figure(figsize=(12, 8))
top_features = importances_series.sort_values().tail(15)
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
top_features.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Mean R² decrease after permutation', fontsize=11)
plt.ylabel('Input features', fontsize=11)
plt.title('Top 15 Most Important Features (Permutation Importance)', fontsize=13, pad=15)
plt.grid(True, linestyle='--', alpha=0.5, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=200)
print("\nFeature importance plot saved as 'results/feature_importance.png'")
plt.show()

importance_df = pd.DataFrame({
    'feature': importances_series.index,
    'importance_mean': importances_series.values,
    'importance_std': importance_results.importances_std
}).sort_values('importance_mean', ascending=False)

importance_df.to_csv('results/feature_importance.csv', index=False)
print("\nFeature importance data saved to 'results/feature_importance.csv'")

print("X_test_proc shape:", X_test_proc.shape)
