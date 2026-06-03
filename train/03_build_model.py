from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import numpy as np, pickle, json, os

os.makedirs('models', exist_ok=True)
os.makedirs('logs',   exist_ok=True)

# ── Primarni model: HistGradientBoosting ─────────────────────────────────────
HGB_HPARAMS = {
    'max_iter':          500,
    'learning_rate':     0.05,
    'max_depth':         6,
    'min_samples_leaf':  20,    # sprečava overfitting na malom datasetu
    'l2_regularization': 0.1,
    'max_leaf_nodes':    31,
    'early_stopping':    True,
    'validation_fraction': 0.1,
    'n_iter_no_change':  20,
    'random_state':      42,
    'verbose':           0,
}

# ── Sekundarni model: MLP (drastično smanjen za dataset od 4k) ───────────────
MLP_HPARAMS = {
    'hidden_layer_sizes': (64, 32),   # ~3000 parametara – razumno za 2884 uzoraka
    'activation':         'relu',
    'solver':             'adam',
    'learning_rate_init': 0.001,
    'alpha':              0.01,       # Jača L2 regularizacija za mali dataset
    'batch_size':         32,
    'max_iter':           1000,
    'random_state':       42,
    'early_stopping':     True,
    'validation_fraction': 0.1,
    'n_iter_no_change':   30,
    'tol':                1e-5,
    'verbose':            False,
}

def build_hgb_model():
    return HistGradientBoostingRegressor(**HGB_HPARAMS)

def build_mlp_model():
    return MLPRegressor(**MLP_HPARAMS)


if __name__ == '__main__':
    try:
        X_train_hgb = np.load('data/X_train_hgb.npy')
        X_train_mlp = np.load('data/X_train_proc.npy')
        input_dim_hgb = X_train_hgb.shape[1]
        input_dim_mlp = X_train_mlp.shape[1]
        n_samples     = X_train_hgb.shape[0]
    except FileNotFoundError:
        print("WARNING: Pokreni 02_preprocessing.py prvo.")
        input_dim_hgb = input_dim_mlp = n_samples = "?"

    # Broj parametara MLP
    if isinstance(n_samples, int):
        mlp_params = (input_dim_mlp*64 + 64) + (64*32 + 32) + (32*1 + 1)
        ratio = n_samples / mlp_params
        print(f"MLP parametri: {mlp_params:,}  |  uzorci/param omjer: {ratio:.2f}")
    
    summary = [
        "=" * 65,
        "PRIMARNI MODEL: HistGradientBoostingRegressor",
        "=" * 65,
        f"Input features    : {input_dim_hgb}",
        f"Max iterations    : {HGB_HPARAMS['max_iter']}  (early stopping)",
        f"Learning rate     : {HGB_HPARAMS['learning_rate']}",
        f"Max depth         : {HGB_HPARAMS['max_depth']}",
        f"Min samples/leaf  : {HGB_HPARAMS['min_samples_leaf']}",
        f"L2 regularization : {HGB_HPARAMS['l2_regularization']}",
        f"Early stopping    : patience={HGB_HPARAMS['n_iter_no_change']}",
        "",
        "=" * 65,
        "SEKUNDARNI MODEL: MLPRegressor (smanjen za mali dataset)",
        "=" * 65,
        f"Input features    : {input_dim_mlp}",
        f"Hidden layers     : (64, 32)  ← drastično smanjen (bio 512,256,128,64)",
        f"Alpha (L2)        : {MLP_HPARAMS['alpha']}",
        f"Max iterations    : {MLP_HPARAMS['max_iter']}",
        "=" * 65,
    ]
    print('\n'.join(summary))

    with open('logs/model_summary.txt', 'w', encoding='utf-8') as f:        f.write('\n'.join(summary) + '\n')
    
    with open('logs/hgb_hparams.json', 'w') as f:
        json.dump(HGB_HPARAMS, f, indent=2)
    with open('logs/mlp_hparams.json', 'w') as f:
        json.dump({k: list(v) if isinstance(v, tuple) else v
                   for k, v in MLP_HPARAMS.items()}, f, indent=2)
    # Backwards compat
    with open('logs/hparams.json', 'w') as f:
        json.dump({k: list(v) if isinstance(v, tuple) else v
                   for k, v in MLP_HPARAMS.items()}, f, indent=2)

    print("\nSaved: logs/model_summary.txt, logs/hgb_hparams.json, logs/mlp_hparams.json")