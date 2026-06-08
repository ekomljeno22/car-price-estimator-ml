import numpy as np
import json
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

def run_evaluation():
    print("\n--- Pokrećem Evaluaciju Modela ---")
    os.makedirs('results', exist_ok=True)

    if not os.path.exists('models/best_model.pkl'):
        print("❌ Greška: 'models/best_model.pkl' ne postoji. Prvo pokreni 'train'.")
        return

    with open('models/best_model.pkl', 'rb') as f:
        bundle = pickle.load(f)

    if isinstance(bundle, dict):
        model      = bundle['model']
        model_name = bundle['type']
    else:
        model      = bundle
        model_name = type(bundle).__name__

    model_class = type(model).__name__
    is_tree = ('Hist' in model_name or 'Gradient' in model_class or 'ExtraTrees' in model_name or 'ExtraTrees' in model_class)
    
    test_path = 'data/X_test_hgb.npy' if is_tree else 'data/X_test_mlp.npy'
    if not os.path.exists(test_path) or not os.path.exists('data/y_test.npy'):
        print(f"❌ Greška: Testne matrice nisu pronađene. Prvo pokreni 'preprocess'.")
        return

    X_test = np.load(test_path)
    y_test = np.load('data/y_test.npy')

    print(f"Evaluiram model: {model_name}")
    print(f"Veličina testne matrice: {X_test.shape}\n")

    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log).clip(min=3000, max=120000)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    print("=" * 65)
    print(f"MODEL PERFORMANCE REPORT: {model_name}")
    print("=" * 65)
    print(f"MAE  : ${mae:>12,.2f}")
    print(f"RMSE : ${rmse:>12,.2f}")
    print(f"R²   :  {r2:>10.4f}")
    print(f"MAPE : {mape:>10.2f}%  {'✓ PASS (≤10%)' if mape <= 10 else '✗ FAIL (>10%)'}")
    print("=" * 65)

    print("\nMAPE po cjenovnim segmentima:")
    for lo, hi, lbl in [(0, 10000, '<$10k'), (10000, 30000, '$10k–$30k'),
                        (30000, 75000, '$30k–$75k'), (75000, np.inf, '>$75k')]:
        m = (y_true >= lo) & (y_true < hi)
        if m.sum() < 5: continue
        seg = mean_absolute_percentage_error(y_true[m], y_pred[m]) * 100
        print(f"  {lbl:<12}: {seg:6.2f}%  (n={m.sum()})")

    metrics = dict(model=model_name, mae=round(float(mae), 2), rmse=round(float(rmse), 2),
                    r2=round(float(r2), 4), mape=round(float(mape), 4))
    with open('results/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nSpremljeno: results/metrics.json")

    cap  = np.percentile(y_true, 99)
    mask = (y_true <= cap) & (y_pred <= cap)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(y_true[mask], y_pred[mask], alpha=0.35, s=18, color='teal', edgecolors='none', label='Predictions')
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
    print("Spremljeno: results/prediction_vs_actual.png")

    pct_errors = np.abs((y_true - y_pred) / y_true.clip(min=1)) * 100
    pct_errors = pct_errors[np.isfinite(pct_errors)]
    within_10  = (pct_errors <= 10).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pct_errors, bins=np.arange(0, min(pct_errors.max() + 5, 105), 2.5), color='royalblue', edgecolor='white', linewidth=0.5)
    ax.axvline(10, color='red', lw=2, linestyle='--', label='10% threshold')
    ax.axvline(mape, color='orange', lw=2, linestyle='-', label=f'MAPE = {mape:.1f}%')
    ax.set_xlabel('Absolute Percentage Error (%)'); ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution  ({within_10:.1f}% predictions within ±10%)')
    ax.legend(); ax.grid(True, linestyle=':', alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/error_distribution.png', dpi=200)
    plt.close()
    print("Spremljeno: results/error_distribution.png")

    residuals = y_true - y_pred
    print(f"\nRezidual analiza:")
    print(f"  Mean    : ${np.mean(residuals):>+12,.2f}")
    print(f"  Std     : ${np.std(residuals):>12,.2f}")
    print(f"  Within ±10% : {within_10:.1f}%")
    print(f"  Within  ±5% : {(pct_errors <= 5).mean() * 100:.1f}%")

    history_path = '../car-price-estimator-ml/logs/training_history.json'
    print(f"\nProvjeravam povijest treniranja na: {os.path.abspath(history_path)}")

    if os.path.exists(history_path):
        print("Datoteka 'training_history.json' uspješno pronađena. Učitavam...")
        with open(history_path) as f:
            history = json.load(f)

        loss_curve = history.get('loss_curve', [])
        print(f"📊 Učitano stavki iz 'loss_curve': {len(loss_curve)}")
        
        if len(loss_curve) > 0:
            fig_lc, ax_lc = plt.subplots(figsize=(7, 5))
            epochs = np.arange(1, len(loss_curve) + 1)

            if len(loss_curve) > 10:
                smooth = np.convolve(loss_curve, np.ones(5)/5, mode='same')
                ax_lc.plot(epochs, loss_curve, color='lightblue', lw=0.8, alpha=0.6, label='Raw loss')
                ax_lc.plot(epochs, smooth,     color='royalblue', lw=2,   label='Smoothed (w=5)')
            else:
                ax_lc.plot(epochs, loss_curve, color='royalblue', lw=2, marker='o', label='Train loss')

            ax_lc.set_xlabel('Epoch')
            ax_lc.set_ylabel('MSE Loss')
            ax_lc.set_title('Training Loss Curve')
            ax_lc.legend()
            ax_lc.grid(True, linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig('results/learning_curves.png', dpi=200)
            plt.close()
            print("Spremljeno: results/learning_curves.png")
        else:
            print("'loss_curve' je prazna unutar datoteke.")
    else:
        print(f"Greška: Datoteka ne postoji na putanji. Trenutna radna mapa skripte je: {os.getcwd()}")