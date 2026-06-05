import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from src.utils import load_config, ensure_directories

def run_preprocessing(df):
    print("\n--- Pokrećem Preprocessing & Feature Engineering ---")
    config = load_config()
    ensure_directories()
    
    df = df.dropna(subset=['price', 'mileage', 'model_year'])
    df = df[(df['price'] >= 3000) & (df['price'] <= 120000)].reset_index(drop=True)
    df = df[df['mileage'] <= 250000].reset_index(drop=True)
    df = df[df['model_year'] >= 2000].reset_index(drop=True)
    
    df['model_age'] = config['current_year'] - df['model_year']
    df['mileage_per_year'] = df['mileage'] / df['model_age'].clip(lower=1)
    df['mileage_factor'] = df['mileage'] * df['model_age']

    df['hp_per_liter'] = df['hp'] / df['liters'].clip(lower=0.5)
    
    for col in ['fuel_type', 'transmission', 'accident', 'clean_title']:
        df[col] = df[col].fillna('Unknown').astype(str).str.strip()
        
    numeric_features = ['model_year', 'model_age', 'mileage', 'mileage_per_year', 'hp', 'liters']
    categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
    high_card_features = ['brand', 'model']
    
    X = df[numeric_features + categorical_features + high_card_features].copy()
    y = np.log1p(df['price'].copy())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_test = X_train.reset_index(drop=True), X_test.reset_index(drop=True)
    
    rare_maps = {}
    for col in ['transmission', 'fuel_type']:
        counts = X_train[col].value_counts()
        rare_categories = set(counts[counts < config['min_category_count']].index)
        rare_maps[col] = rare_categories
        X_train[col] = X_train[col].apply(lambda x: 'Other' if x in rare_categories else x)
        X_test[col] = X_test[col].apply(lambda x: 'Other' if x in rare_categories else x)

    KF = KFold(n_splits=5, shuffle=True, random_state=42)
    global_mean = float(y_train.mean())
    te_maps = {'global_mean': global_mean, 'smoothing': config['te_smoothing']}
    
    for col in high_card_features:
        X_train[col + '_te'] = global_mean
        for tr_idx, val_idx in KF.split(X_train):
            tr_fold, y_fold = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
            stats = y_fold.groupby(tr_fold[col]).agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + global_mean * config['te_smoothing']) / (stats['count'] + config['te_smoothing'])
            X_train.loc[val_idx, col + '_te'] = X_train.iloc[val_idx][col].map(smooth).fillna(global_mean).values
            
        full_stats = y_train.groupby(X_train[col]).agg(['mean', 'count'])
        full_map = (full_stats['mean'] * full_stats['count'] + global_mean * config['te_smoothing']) / (full_stats['count'] + config['te_smoothing'])
        te_maps[col] = full_map
        X_test[col + '_te'] = X_test[col].map(full_map).fillna(global_mean)
        
        X_train.drop(columns=[col], inplace=True)
        X_test.drop(columns=[col], inplace=True)
        
    X_train.rename(columns={c + '_te': c for c in high_card_features}, inplace=True)
    X_test.rename(columns={c + '_te': c for c in high_card_features}, inplace=True)
    
    brand_model_map = df.groupby("brand")["model"].apply(lambda s: sorted(s.unique().tolist())).to_dict()
    te_maps["brand_model_map"] = brand_model_map
    
    ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train_cat_enc = ord_enc.fit_transform(X_train[categorical_features]).astype(int)
    X_test_cat_enc = ord_enc.transform(X_test[categorical_features]).astype(int)
    
    num_and_te = numeric_features + high_card_features
    X_train_hgb = np.hstack([X_train[num_and_te].values, X_train_cat_enc])
    X_test_hgb = np.hstack([X_test[num_and_te].values, X_test_cat_enc])
    cat_indices = list(range(len(num_and_te), len(num_and_te) + len(categorical_features)))
    
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    scaler = StandardScaler()
    X_train_mlp = np.hstack([scaler.fit_transform(X_train[num_and_te].values), ohe.fit_transform(X_train[categorical_features])])
    X_test_mlp = np.hstack([scaler.transform(X_test[num_and_te].values), ohe.transform(X_test[categorical_features])])
    
    np.save('data/X_train_hgb.npy', X_train_hgb)
    np.save('data/X_test_hgb.npy', X_test_hgb)
    np.save('data/X_train_mlp.npy', X_train_mlp)
    np.save('data/X_test_mlp.npy', X_test_mlp)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    
    with open('data/te_maps.pkl', 'wb') as f: pickle.dump(te_maps, f)
    with open('data/rare_maps.pkl', 'wb') as f: pickle.dump(rare_maps, f)
    with open('data/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
    with open('data/ohe.pkl', 'wb') as f: pickle.dump(ohe, f)
    with open('data/ord_enc.pkl', 'wb') as f: pickle.dump(ord_enc, f)
    
    meta = {
        'categorical_feature_indices': cat_indices,
        'feature_names_hgb': num_and_te + categorical_features,
        'feature_names_mlp': num_and_te + list(ohe.get_feature_names_out(categorical_features)),
        'current_year': config['current_year']
    }
    with open('data/pipeline_meta.pkl', 'wb') as f: pickle.dump(meta, f)
    print("✅ Preprocessing završen. Podaci očišćeni od ekstremnih vrijednosti.")