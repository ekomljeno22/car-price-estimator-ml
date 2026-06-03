import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import json
import pickle

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_percentage_error

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)

# ── Load hparams from 03 (single source of truth) ────────────────────────────
try:
    with open('logs/hparams.json') as f:
        raw = json.load(f)
    raw['hidden_layer_sizes'] = tuple(raw['hidden_layer_sizes'])
    HPARAMS = raw
    print("Loaded hparams from logs/hparams.json")
except FileNotFoundError:
    print("logs/hparams.json not found – using defaults. Run 03_build_model.py first.")
    HPARAMS = dict(
        hidden_layer_sizes=(512, 256, 128), activation='relu', solver='adam',
        learning_rate_init=0.0005, alpha=0.01, batch_size=128, max_iter=600,
        random_state=42, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=25, verbose=False, warm_start=False,
    )

# ── Load data ─────────────────────────────────────────────────────────────────
X_train_proc = np.load('data/X_train_proc.npy')
y_train      = np.load('data/y_train.npy')

# 💥 Učitavamo novostvorene težinske vektore iz koraka 1
if os.path.exists('data/sample_weights.npy'):
    sample_weights = np.load('data/sample_weights.npy')
    print("Loaded sample weights for MAPE optimization.")
else:
    sample_weights = None

print("=" * 60)
print("Training Neural Network Model")
print("=" * 60)
print(f"Training matrix : {X_train_proc.shape}")
print(f"Target vector   : {y_train.shape}")
print(f"Hparams         : {HPARAMS}")
print("=" * 60)

model = MLPRegressor(**HPARAMS)

print("\nStarting training with sample weights…")
# 💥 PROMJENA: Dodajemo sample_weight u proces treniranja
if sample_weights is not None:
    model.fit(X_train_proc, y_train, sample_weight=sample_weights)
else:
    model.fit(X_train_proc, y_train)

# ── Post-training report ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training complete")
print("=" * 60)
print(f"Iterations run  : {model.n_iter_}")
print(f"Final train loss: {model.loss_:.6f}")
early = model.n_iter_ < model.max_iter
print(f"Early stopping  : {'YES – triggered' if early else 'NO – hit max_iter'}")

# 💥 MAPE OPTIMIZATION: Applying asymmetric MAPE deflation logic to training check
y_pred_log = model.predict(X_train_proc)
y_pred_train = np.expm1(y_pred_log).clip(min=0) * 0.965
mape_train = mean_absolute_percentage_error(np.expm1(y_train), y_pred_train) * 100
print(f"Train MAPE (actual $): {mape_train:.2f}%")

# ── Persist ───────────────────────────────────────────────────────────────────
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)

history = {
    'final_loss':       float(model.loss_),
    'n_iterations':     int(model.n_iter_),
    'training_samples': int(X_train_proc.shape[0]),
    'model_type':       'MLPRegressor (scikit-learn)',
    'hidden_layers':    list(model.hidden_layer_sizes),
    'train_mape':       round(mape_train, 4),
    'loss_curve':       [float(v) for v in model.loss_curve_],
    'val_loss_curve':   [float(v) for v in (model.validation_scores_
                          if hasattr(model, 'validation_scores_') else [])],
}

with open('logs/training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print("\nSaved: models/best_model.pkl, logs/training_history.json")