from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import numpy as np
import pandas as pd
import pickle, os

os.makedirs('data', exist_ok=True)

# ── Load & parse ──────────────────────────────────────────────────────────────
df = pd.read_csv('used_cars.csv')

df['price']  = pd.to_numeric(df['price'].str.replace(r'[$,]',   '', regex=True), errors='coerce')
df['milage'] = pd.to_numeric(df['milage'].str.replace(r'[^\d.]', '', regex=True), errors='coerce')

df = df.dropna(subset=['price', 'milage', 'model_year'])
df = df[df['price'] >= 2500].reset_index(drop=True)

print(f"Rows after price filter (>=$2500): {len(df)}")

# ── IQR clipping ──────────────────────────────────────────────────────────────
def iqr_clip(series, factor=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return series.clip(lower=q1 - factor*(q3-q1), upper=q3 + factor*(q3-q1))

df['price']  = iqr_clip(df['price'])
df['milage'] = iqr_clip(df['milage'])

# ── Feature engineering ───────────────────────────────────────────────────────
CURRENT_YEAR   = 2025
df['model_age']      = CURRENT_YEAR - df['model_year']
df['milage_log']     = np.log1p(df['milage'])
df['miles_per_year'] = df['milage'] / df['model_age'].clip(lower=1)

# Fillna za kategoričke
for col in ['fuel_type', 'transmission', 'accident', 'clean_title']:
    df[col] = df[col].fillna('Unknown')

numeric_features     = ['model_year', 'model_age', 'milage_log', 'miles_per_year']
categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
high_card_features   = ['brand', 'model']

X = df[numeric_features + categorical_features + high_card_features].copy()
y = np.log1p(df['price'])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# ── FIX: Grupiranje rijetkih kategorija (Bez Data Leakage-a) ─────────────────
MIN_CATEGORY_COUNT = 20
rare_maps = {}

for col in ['transmission', 'fuel_type']:
    # Izračunaj frekvencije ISKLJUČIVO na X_train
    counts = X_train[col].value_counts()
    rare_categories = set(counts[counts < MIN_CATEGORY_COUNT].index)
    rare_maps[col] = rare_categories
    
    # Primijeni na Train i Test
    X_train[col] = X_train[col].apply(lambda x: 'Other' if x in rare_categories else x)
    X_test[col]  = X_test[col].apply(lambda x: 'Other' if x in rare_categories else x)
    print(f"{col}: {len(counts)} → {X_train[col].nunique()} kategorija u treningu")

# ── TARGET ENCODING u log-prostoru (KFold OOF) ───────────────────────────────
KF          = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = float(y_train.mean())
te_maps     = {'global_mean': global_mean}

for col in high_card_features:
    X_train[col + '_te'] = global_mean

    for tr_idx, val_idx in KF.split(X_train):
        fold_map = y_train.iloc[tr_idx].groupby(X_train.iloc[tr_idx][col].values).mean()
        X_train.loc[val_idx, col + '_te'] = (
            X_train.iloc[val_idx][col].map(fold_map).fillna(global_mean).values
        )

    full_map     = y_train.groupby(X_train[col]).mean()
    te_maps[col] = full_map
    X_test[col + '_te'] = X_test[col].map(full_map).fillna(global_mean)

    X_train.drop(columns=[col], inplace=True)
    X_test.drop( columns=[col], inplace=True)

X_train.rename(columns={c + '_te': c for c in high_card_features}, inplace=True)
X_test.rename( columns={c + '_te': c for c in high_card_features}, inplace=True)

# ── PIPELINE 1: Za HistGradientBoosting (OrdinalEncoder) ──────────────────────
ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

X_train_cat = X_train[categorical_features].copy()
X_test_cat  = X_test[categorical_features].copy()

X_train_cat_enc = ord_enc.fit_transform(X_train_cat).astype(int)
X_test_cat_enc  = ord_enc.transform(X_test_cat).astype(int)

num_and_te = numeric_features + high_card_features
X_train_hgb = np.hstack([X_train[num_and_te].values, X_train_cat_enc])
X_test_hgb  = np.hstack([X_test[num_and_te].values,  X_test_cat_enc])

n_num = len(num_and_te)
categorical_feature_indices = list(range(n_num, n_num + len(categorical_features)))

print(f"\nHGB matrix shape – train: {X_train_hgb.shape}, test: {X_test_hgb.shape}")

# ── PIPELINE 2: Za MLP (StandardScaler + OHE) ─────────────────────────────────
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = StandardScaler()

X_train_num_te = scaler.fit_transform(X_train[num_and_te].values)
X_test_num_te  = scaler.transform(X_test[num_and_te].values)

X_train_ohe = ohe.fit_transform(X_train[categorical_features])
X_test_ohe  = ohe.transform(X_test[categorical_features])

X_train_mlp = np.hstack([X_train_num_te, X_train_ohe])
X_test_mlp  = np.hstack([X_test_num_te,  X_test_ohe])

print(f"MLP matrix shape  – train: {X_train_mlp.shape}, test: {X_test_mlp.shape}")

# ── Persist ───────────────────────────────────────────────────────────────────
np.save('data/X_train_hgb.npy', X_train_hgb)
np.save('data/X_test_hgb.npy',  X_test_hgb)
np.save('data/X_train_proc.npy', X_train_mlp)
np.save('data/X_test_proc.npy',  X_test_mlp)
np.save('data/y_train.npy', y_train.values)
np.save('data/y_test.npy',  y_test.values)

with open('data/te_maps.pkl', 'wb') as f:
    pickle.dump(te_maps, f)

# Sačuvaj i mape za rijetke kategorije kako bi ih produkcijski pipeline prepoznao!
with open('data/rare_maps.pkl', 'wb') as f:
    pickle.dump(rare_maps, f)

pipeline_meta = {
    'categorical_feature_indices': categorical_feature_indices,
    'feature_names_hgb': num_and_te + categorical_features,
    'n_num_features': n_num,
}
with open('data/pipeline_meta.pkl', 'wb') as f:
    pickle.dump(pipeline_meta, f)

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('data/ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)
with open('data/ord_enc.pkl', 'wb') as f:
    pickle.dump(ord_enc, f)

print("\nSve datoteke uspješno spremljene u 'data/' direktorij!")