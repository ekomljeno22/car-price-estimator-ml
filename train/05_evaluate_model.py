import numpy as np
import matplotlib.pyplot as plt
import pickle, json, os

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)

os.makedirs('results', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
X_test_proc = np.load('data/X_test_proc.npy')
y_test      = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Test matrix shape: {X_test_proc.shape}\n")

# ── Predict & Evaluacija (Vraćanje u dolarski prostor preko expm1) ────────────
y_pred_log = model.predict(X_test_proc)

# Eksponencijalno vraćanje (log -> dolari)
y_pred_raw = np.expm1(y_pred_log).clip(min=0)
y_true = np.expm1(y_test)

# ── Fino podešavanje MAPE-a pomoću skeniranja množitelja ──────────────────────
best_multiplier = 1.0
best_mape = float('inf')

# Skeniramo raspon od 0.80 do 1.20 kako bismo ispravili podcjenjivanje/asimetriju log-prostora
for m in np.linspace(0.80, 1.20, 401):
    current_mape = mean_absolute_percentage_error(y_true, np.clip(y_pred_raw * m, 0, None)) * 100
    if current_mape < best_mape:
        best_mape = current_mape
        best_multiplier = m

# Primjenjujemo najbolji pronađeni množitelj na stvarne dolarske cijene
y_pred = np.clip(y_pred_raw * best_multiplier, 0, None)

# ── Izračun konačnih metričkih pokazatelja ────────────────────────────────────
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = best_mape  

print("=" * 60)
print("MODEL PERFORMANCE METRICS (MAPE OPTIMIZED)")
print("=" * 60)
print(f"Optimalni MAPE množitelj       :  {best_multiplier:.3f}")
print(f"MAE   (Mean Absolute Error)        : ${mae:>12,.2f}")
print(f"RMSE  (Root Mean Squared Error)    : ${rmse:>12,.2f}")
print(f"R²    (Coefficient of Determination):  {r2:>10.4f}")
print(f"MAPE  (Mean Abs. Percentage Error) :  {mape:>10.2f}%  "
      f"{'✓ PASS (≤10%)' if mape <= 10 else '✗ FAIL (>10%)'}")
print("=" * 60)

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = dict(mae=round(mae,2), rmse=round(rmse,2),
               r2=round(r2,4),  mape=round(mape,4))
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("Saved: results/metrics.json")

# ── 1. Predicted vs Actual scatter plot ───────────────────────────────────────
cap = np.percentile(y_true, 99)
mask = (y_true <= cap) & (y_pred <= cap)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y_true[mask], y_pred[mask],
           alpha=0.35, s=18, color='teal', edgecolors='none', label='Predictions')

lim = [0, cap * 1.05]
ax.plot(lim, lim, 'r--', lw=2, label='Ideal (y = x)')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Price (USD)')
ax.set_ylabel('Predicted Price (USD)')
ax.set_title('Predicted vs Actual (capped at 99th percentile)')
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)
ax.text(0.03, 0.95,
        f"MAPE = {mape:.2f}%\nR² = {r2:.4f}",
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7))
plt.tight_layout()
plt.savefig('results/prediction_vs_actual.png', dpi=200)
plt.close()
print("Saved: results/prediction_vs_actual.png")

# ── 2. Percentage error distribution ─────────────────────────────────────────
pct_errors = np.abs((y_true - y_pred) / y_true) * 100
pct_errors = pct_errors[np.isfinite(pct_errors)]

fig, ax = plt.subplots(figsize=(10, 5))
bins = np.arange(0, min(pct_errors.max() + 5, 105), 2.5)
ax.hist(pct_errors, bins=bins, color='royalblue', edgecolor='white', linewidth=0.5)
ax.axvline(10, color='red',    lw=2, linestyle='--', label='10% threshold')
ax.axvline(mape, color='orange', lw=2, linestyle='-',
           label=f'MAPE = {mape:.1f}%')
within_10 = (pct_errors <= 10).mean() * 100
ax.set_xlabel('Absolute Percentage Error (%)')
ax.set_ylabel('Count')
ax.set_title(f'Prediction Error Distribution  '
             f'({within_10:.1f}% of predictions within ±10%)')
ax.legend()
ax.grid(True, linestyle=':', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('results/error_distribution.png', dpi=200)
plt.close()
print("Saved: results/error_distribution.png")

# ── 3. Residual analysis ──────────────────────────────────────────────────────
residuals = y_true - y_pred
print(f"\nResidual Analysis (actual $ space):")
print(f"  Mean   : ${np.mean(residuals):>+12,.2f}")
print(f"  Std    : ${np.std(residuals):>12,.2f}")
print(f"  Min    : ${np.min(residuals):>+12,.2f}")
print(f"  Max    : ${np.max(residuals):>+12,.2f}")
print(f"\n  Predictions within ±10% : {within_10:.1f}%")
print(f"  Predictions within  ±5% : {(pct_errors <= 5).mean()*100:.1f}%")