import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import json, os

os.makedirs('results', exist_ok=True)

# ── Load training history ─────────────────────────────────────────────────────
with open('logs/training_history.json') as f:
    history = json.load(f)

loss_curve = history.get('loss_curve', [])
val_scores = history.get('val_loss_curve', [])   # validation R² per epoch

# Load test metrics if available
try:
    with open('results/metrics.json') as f:
        metrics = json.load(f)
except FileNotFoundError:
    metrics = {}

# ── Figure layout ─────────────────────────────────────────────────────────────
has_val   = len(val_scores) > 0
n_panels  = 3 if has_val else 2
fig = plt.figure(figsize=(6 * n_panels, 5))
gs  = gridspec.GridSpec(1, n_panels, figure=fig)

# ── Panel 1: Training loss curve ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
epochs = np.arange(1, len(loss_curve) + 1)

if len(loss_curve) > 10:
    # Light smoothing (rolling mean, window=5) to show trend
    smooth = np.convolve(loss_curve, np.ones(5)/5, mode='same')
    ax1.plot(epochs, loss_curve, color='lightblue', lw=0.8, alpha=0.6, label='Raw loss')
    ax1.plot(epochs, smooth,     color='royalblue', lw=2,   label='Smoothed (w=5)')
else:
    ax1.plot(epochs, loss_curve, color='royalblue', lw=2, marker='o', label='Train loss')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('MSE Loss')
ax1.set_title('Training Loss Curve')
ax1.legend()
ax1.grid(True, linestyle=':', alpha=0.5)

# ── Panel 2 (optional): Validation R² curve ──────────────────────────────────
panel_idx = 1
if has_val:
    ax2 = fig.add_subplot(gs[panel_idx])
    val_epochs = np.arange(1, len(val_scores) + 1)
    ax2.plot(val_epochs, val_scores, color='darkorange', lw=2, label='Val R²')
    ax2.axhline(0, color='grey', lw=0.8, linestyle='--')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('R²')
    ax2.set_title('Validation R² per Epoch')
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.5)
    panel_idx += 1

# ── Panel 3: Summary text ─────────────────────────────────────────────────────
ax_s = fig.add_subplot(gs[panel_idx])
ax_s.axis('off')

mape_val  = metrics.get('mape',  history.get('train_mape', 'N/A'))
mape_str  = f"{mape_val:.2f}%" if isinstance(mape_val, float) else str(mape_val)
mape_pass = isinstance(mape_val, float) and mape_val <= 10

summary = [
    ('Final train loss',  f"{history.get('final_loss', 'N/A'):.6f}"),
    ('Epochs ran',        str(history.get('n_iterations', 'N/A'))),
    ('Training samples',  f"{history.get('training_samples', 'N/A'):,}"),
    ('Hidden layers',     str(history.get('hidden_layers', 'N/A'))),
    ('',                  ''),
    ('Test MAPE',         mape_str + (' ✓' if mape_pass else ' ✗ >10%')),
    ('Test R²',           f"{metrics.get('r2', 'N/A')}"),
    ('Test MAE',          f"${metrics.get('mae', 'N/A'):,.0f}" if isinstance(metrics.get('mae'), float) else 'N/A'),
    ('Test RMSE',         f"${metrics.get('rmse','N/A'):,.0f}" if isinstance(metrics.get('rmse'), float) else 'N/A'),
]

y_pos = 0.95
ax_s.text(0.5, y_pos + 0.04, 'Training & Test Summary',
          ha='center', fontsize=13, fontweight='bold', transform=ax_s.transAxes)
for label, value in summary:
    if not label:
        y_pos -= 0.04
        continue
    color = ('green' if mape_pass else 'red') if 'MAPE' in label else 'black'
    ax_s.text(0.05, y_pos, f"{label}:", fontsize=10, transform=ax_s.transAxes)
    ax_s.text(0.62, y_pos, value,       fontsize=10, transform=ax_s.transAxes,
              color=color, fontweight='bold' if 'MAPE' in label else 'normal')
    y_pos -= 0.09

plt.suptitle('Model Training Report', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/learning_curves.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: results/learning_curves.png")

# Also print to console
print("\n" + "=" * 50)
print("TRAINING & TEST SUMMARY")
print("=" * 50)
for label, value in summary:
    if label:
        print(f"  {label:<22}: {value}")
print("=" * 50)