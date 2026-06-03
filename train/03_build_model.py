from sklearn.neural_network import MLPRegressor
import numpy as np
import pickle
import json
import os

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)

# ── Hyperparameters (single source of truth shared with 04_train_model.py) ────
# 💥 MAPE OPTIMIZATION: Widened layers, slowed learning rate, and added heavy L2 penalty
# ── Unutar 03_build_model.py ──────────────────────────────────────────────────

HPARAMS = {
    'hidden_layer_sizes': (512, 256, 128),  # Zadržavamo stabilnu širinu slojeva
    'activation':         'relu',
    'solver':             'adam',
    'learning_rate_init': 0.001,           # Vraćamo na standardnu brzinu za bolji fokus
    'alpha':              0.001,           # Smanjujemo s 0.01 na 0.001 (manje agresivna kazna)
    'batch_size':         64,              # Manji batch omogućuje češće korekcije težina
    'max_iter':           500,
    'random_state':       42,
    'early_stopping':     True,
    'validation_fraction':0.1,
    'n_iter_no_change':   20,
    'verbose':            False,
    'warm_start':         False,
}

def build_model() -> MLPRegressor:
    return MLPRegressor(**HPARAMS)


if __name__ == '__main__':
    # Validate against saved data
    try:
        X_train_proc = np.load('data/X_train_proc.npy')
        input_dim    = X_train_proc.shape[1]
    except FileNotFoundError:
        print("WARNING: data/X_train_proc.npy not found – run 02_preprocessing.py first.")
        input_dim = "?"

    summary_lines = [
        "=" * 60,
        "MLP Regression Model Architecture (Optimized for MAPE)",
        "=" * 60,
        f"Input dimension  : {input_dim}",
        "",
        "Hidden Layer 1   : 512 neurons  (ReLU)",
        "Hidden Layer 2   : 256 neurons  (ReLU)",
        "Hidden Layer 3   : 128 neurons  (ReLU)",
        "Output Layer     :   1 neuron   (Linear – regression)",
        "",
        f"Optimizer        : Adam  (lr={HPARAMS['learning_rate_init']})",
        f"Loss             : MSE",
        f"Regularisation   : L2   (alpha={HPARAMS['alpha']})",
        f"Batch size       : {HPARAMS['batch_size']}",
        f"Max iterations   : {HPARAMS['max_iter']}",
        f"Early stopping   : patience={HPARAMS['n_iter_no_change']} epochs,  "
        f"val={int(HPARAMS['validation_fraction']*100)}%",
        "=" * 60,
    ]

    print('\n'.join(summary_lines))

    with open('logs/model_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    with open('logs/hparams.json', 'w') as f:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in HPARAMS.items()}, f, indent=2)

    # Persist untrained model skeleton
    model = build_model()
    with open('models/model_object.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("\nSaved: logs/model_summary.txt, logs/hparams.json, models/model_object.pkl")