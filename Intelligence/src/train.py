import numpy as np
import pickle
import os
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from src.utils import load_config

def run_training():
    print("\n--- Pokrećem Treniranje Modela s Optimizacijom ---")
    os.makedirs('models', exist_ok=True)
    config = load_config()
    
    X_train_hgb = np.load('data/X_train_hgb.npy')
    X_train_mlp = np.load('data/X_train_mlp.npy')
    y_train = np.load('data/y_train.npy')
    
    with open('data/pipeline_meta.pkl', 'rb') as f:
        meta = pickle.load(f)
        
    print(f"Uzoraka za trening: {X_train_hgb.shape[0]}")
    
    y_train_dollars = np.expm1(y_train).clip(min=1000)
    sample_weights = 1.0 / y_train_dollars
    sample_weights = sample_weights / np.mean(sample_weights)
    
    hgb = HistGradientBoostingRegressor(
        categorical_features=meta['categorical_feature_indices'], 
        **config['hgb_hparams']
    )
    mlp = MLPRegressor(**config['mlp_hparams'])
    et = ExtraTreesRegressor(**config['et_hparams'])
    
    print("-> Treniram HistGradientBoosting (s težinama)...")
    hgb.fit(X_train_hgb, y_train, sample_weight=sample_weights)
    
    print("-> Treniram MLP Neuralnu Mrežu...")
    mlp.fit(X_train_mlp, y_train)
    
    print("-> Treniram ExtraTrees (s težinama)...")
    et.fit(X_train_hgb, y_train, sample_weight=sample_weights)
    
    with open('models/hgb_model.pkl', 'wb') as f: pickle.dump(hgb, f)
    with open('models/mlp_model.pkl', 'wb') as f: pickle.dump(mlp, f)
    with open('models/extra_trees_model.pkl', 'wb') as f: pickle.dump(et, f)
    
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump({'model': hgb, 'type': 'HistGradientBoosting'}, f)
        
    print("✅ Modeli istrenirani. Težine su primijenjene za balansiranje greške.")