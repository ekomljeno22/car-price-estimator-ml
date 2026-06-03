import numpy as np, json, pickle, os
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error, r2_score

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)

# ── Učitaj podatke ─────────────────────────────────────────────────────────────
X_train_hgb = np.load('data/X_train_hgb.npy')
X_train_mlp = np.load('data/X_train_proc.npy')
y_train     = np.load('data/y_train.npy')

with open('data/pipeline_meta.pkl', 'rb') as f:
    meta = pickle.load(f)
cat_indices = meta['categorical_feature_indices']

# ── Učitaj hparamove ──────────────────────────────────────────────────────────
with open('logs/hgb_hparams.json') as f:
    HGB_HPARAMS = json.load(f)
with open('logs/mlp_hparams.json') as f:
    raw = json.load(f)
    raw['hidden_layer_sizes'] = tuple(raw['hidden_layer_sizes'])
    MLP_HPARAMS = raw

# ── 1. Treniranje HGB modela ──────────────────────────────────────────────────
print("=" * 65)
print("Treniranje HistGradientBoostingRegressor (primarni model)")
print("=" * 65)
print(f"   Train matrix : {X_train_hgb.shape}")
print(f"   Kategorički indeksi: {cat_indices}")

hgb = HistGradientBoostingRegressor(
    categorical_features=cat_indices,
    **HGB_HPARAMS
)
hgb.fit(X_train_hgb, y_train)

# Dohvaćanje predikcija i inverzija logaritma
y_true_train = np.expm1(y_train)
y_pred_hgb   = np.expm1(hgb.predict(X_train_hgb)).clip(min=0)

mape_hgb     = mean_absolute_percentage_error(y_true_train, y_pred_hgb) * 100
r2_hgb       = r2_score(y_true_train, y_pred_hgb)  # <-- POPRAVLJENO: proslijeđen y_pred_hgb u dolarima

print(f"   Iteracije   : {hgb.n_iter_}")
print(f"   Train MAPE  : {mape_hgb:.2f}%")
print(f"   Train R²    : {r2_hgb:.4f}")  # <-- Ovo će sada biti visoko i pozitivno!

with open('models/hgb_model.pkl', 'wb') as f:
    pickle.dump(hgb, f)
print("   Saved: models/hgb_model.pkl")

# ── 2. Treniranje MLP modela ──────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Treniranje MLPRegressor (sekundarni model, smanjen)")
print("=" * 65)
print(f"   Train matrix : {X_train_mlp.shape}")

mlp = MLPRegressor(**MLP_HPARAMS)
mlp.fit(X_train_mlp, y_train)

# Dohvaćanje predikcija i inverzija logaritma
y_pred_mlp = np.expm1(mlp.predict(X_train_mlp)).clip(min=0)

mape_mlp   = mean_absolute_percentage_error(y_true_train, y_pred_mlp) * 100
r2_mlp     = r2_score(y_true_train, y_pred_mlp)  # <-- POPRAVLJENO: proslijeđen y_pred_mlp u dolarima

print(f"   Iteracije   : {mlp.n_iter_}")
print(f"   Train MAPE  : {mape_mlp:.2f}%")
print(f"   Train R²    : {r2_mlp:.4f}")

with open('models/mlp_model.pkl', 'wb') as f:
    pickle.dump(mlp, f)
print("   Saved: models/mlp_model.pkl")

# ── Odabir boljeg modela ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
if mape_hgb <= mape_mlp:
    best_name = 'HistGradientBoosting'
    with open('models/hgb_model.pkl', 'rb') as f:
        best = pickle.load(f)
else:
    best_name = 'MLP'
    with open('models/mlp_model.pkl', 'rb') as f:
        best = pickle.load(f)

print(f"Bolji model: {best_name}  (MAPE HGB={mape_hgb:.2f}%, MLP={mape_mlp:.2f}%)")
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump({'model': best, 'type': best_name}, f)

# ── Spremi history ────────────────────────────────────────────────────────────
history = {
    'best_model':        best_name,
    'hgb_n_iterations':  int(hgb.n_iter_),
    'hgb_train_mape':    round(float(mape_hgb), 4),
    'hgb_train_r2':      round(float(r2_hgb), 4),
    'mlp_n_iterations':  int(mlp.n_iter_),
    'mlp_final_loss':    float(mlp.loss_),
    'mlp_train_mape':    round(float(mape_mlp), 4),
    'mlp_train_r2':      round(float(r2_mlp), 4),
    'training_samples':  int(X_train_hgb.shape[0]),
    'loss_curve':        [float(v) for v in mlp.loss_curve_],
    'val_loss_curve':    [float(v) for v in getattr(mlp, 'validation_scores_', [])],
    # Backwards compat polja za 06_visualize_curves.py
    'final_loss':        float(mlp.loss_),
    'n_iterations':      int(hgb.n_iter_),
    'model_type':        f'HistGBM + MLP, best={best_name}',
    'hidden_layers':     list(MLP_HPARAMS['hidden_layer_sizes']),
    'train_mape':        round(float(min(mape_hgb, mape_mlp)), 4),
}
with open('logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nSaved: models/best_model.pkl, logs/training_history.json")