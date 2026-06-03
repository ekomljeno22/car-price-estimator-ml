import numpy as np
import matplotlib.pyplot as plt
import pickle, json, os
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)

os.makedirs('results', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
y_test = np.load('data/y_test.npy')

with open('models/best_model.pkl', 'rb') as f:
    bundle = pickle.load(f)

# Podrška za stari format (direktan model) i novi (dict)
if isinstance(bundle, dict):
    model      = bundle['model']
    model_name = bundle['type']
else:
    model      = bundle
    model_name = type(bundle).__name__

# Odaberi ispravan X_test ovisno o modelu
is_hgb = 'Hist' in model_name or 'Gradient' in type(model).__name__
X_test = np.load('data/X_test_hgb.npy' if is_hgb else 'data/X_test_proc.npy')

print(f"Evaluiram: {model_name}")
print(f"Test matrix shape: {X_test.shape}\n")

# ── Predict ───────────────────────────────────────────────────────────────────
y_pred_log = model.predict(X_test)
y_pred     = np.expm1(y_pred_log).clip(min=0)
y_true     = np.expm1(y_test)

# ── Metrike ───────────────────────────────────────────────────────────────────
mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print("=" * 65)
print(f"MODEL: {model_name}")
print("=" * 65)
print(f"MAE   : ${mae:>12,.2f}")
print(f"RMSE  : ${rmse:>12,.2f}")
print(f"R²    :  {r2:>10.4f}")
print(f"MAPE  :  {mape:>10.2f}%  {'✓ PASS (≤10%)' if mape <= 10 else '✗ FAIL (>10%)'}")
print("=" * 65)

# ── MAPE po cjenovnim segmentima ──────────────────────────────────────────────
print("\nMAPE po cjenovnim segmentima:")
for lo, hi, lbl in [(0,10000,'<$10k'),(10000,30000,'$10k–$30k'),
                    (30000,75000,'$30k–$75k'),(75000,np.inf,'>$75k')]:
    m = (y_true >= lo) & (y_true < hi)
    if m.sum() < 5: continue
    seg = mean_absolute_percentage_error(y_true[m], y_pred[m]) * 100
    print(f"  {lbl:<12}: {seg:6.2f}%  (n={m.sum()})")

# ── Spremi metrike ────────────────────────────────────────────────────────────
metrics = dict(model=model_name, mae=round(float(mae),2), rmse=round(float(rmse),2),
               r2=round(float(r2),4), mape=round(float(mape),4))
with open('results/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved: results/metrics.json")

# ── Plot 1: Predicted vs Actual ───────────────────────────────────────────────
cap  = np.percentile(y_true, 99)
mask = (y_true <= cap) & (y_pred <= cap)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y_true[mask], y_pred[mask], alpha=0.35, s=18,
           color='teal', edgecolors='none', label='Predictions')
lim = [0, cap * 1.05]
ax.plot(lim, lim, 'r--', lw=2, label='Ideal (y = x)')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Actual Price (USD)'); ax.set_ylabel('Predicted Price (USD)')
ax.set_title(f'Predicted vs Actual – {model_name}')
fmt = plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.legend(); ax.grid(True, linestyle=':', alpha=0.5)
ax.text(0.03, 0.95, f"MAPE = {mape:.2f}%\nR² = {r2:.4f}",
        transform=ax.transAxes, fontsize=11, va='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
plt.tight_layout()
plt.savefig('results/prediction_vs_actual.png', dpi=200)
plt.close()
print("Saved: results/prediction_vs_actual.png")

# ── Plot 2: Error distribution ────────────────────────────────────────────────
pct_errors = np.abs((y_true - y_pred) / y_true.clip(min=1)) * 100
pct_errors = pct_errors[np.isfinite(pct_errors)]
within_10  = (pct_errors <= 10).mean() * 100

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pct_errors, bins=np.arange(0, min(pct_errors.max()+5, 105), 2.5),
        color='royalblue', edgecolor='white', linewidth=0.5)
ax.axvline(10,   color='red',    lw=2, linestyle='--', label='10% threshold')
ax.axvline(mape, color='orange', lw=2, linestyle='-',  label=f'MAPE = {mape:.1f}%')
ax.set_xlabel('Absolute Percentage Error (%)')
ax.set_ylabel('Count')
ax.set_title(f'Error Distribution  ({within_10:.1f}% predictions within ±10%)')
ax.legend(); ax.grid(True, linestyle=':', alpha=0.4, axis='y')
plt.tight_layout()
plt.savefig('results/error_distribution.png', dpi=200)
plt.close()
print("Saved: results/error_distribution.png")

# ── Residual summary ──────────────────────────────────────────────────────────
residuals = y_true - y_pred
print(f"\nRezidual analiza:")
print(f"  Mean   : ${np.mean(residuals):>+12,.2f}")
print(f"  Std    : ${np.std(residuals):>12,.2f}")
print(f"  Within ±10% : {within_10:.1f}%")
print(f"  Within  ±5% : {(pct_errors<=5).mean()*100:.1f}%")