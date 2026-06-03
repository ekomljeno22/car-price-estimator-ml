from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import pickle
import os

os.makedirs('data', exist_ok=True)

# ── Load & parse ──────────────────────────────────────────────────────────────
df = pd.read_csv('used_cars.csv')

df['price']  = pd.to_numeric(df['price'].str.replace(r'[$,]',   '', regex=True), errors='coerce')
df['milage'] = pd.to_numeric(df['milage'].str.replace(r'[^\d.]', '', regex=True), errors='coerce')

df = df.dropna(subset=['price', 'milage', 'model_year'])

# 💥 MAPE POPRAVAK 1: Filtriramo ekstremno jeftine automobile koji uništavaju nazivnik
df = df[df['price'] >= 2500]

# ── IQR clipping ──────────────────────────────────────────────────────────────
def iqr_clip(series, factor=3.0):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(lower=q1 - factor * iqr, upper=q3 + factor * iqr)

df['price']  = iqr_clip(df['price'])
df['milage'] = iqr_clip(df['milage'])

# nakon filtriranja i clippinga, resetiramo indekse da spriječimo IndexError
df = df.reset_index(drop=True)

# ── Feature engineering ───────────────────────────────────────────────────────
CURRENT_YEAR = 2026
df['model_age']  = CURRENT_YEAR - df['model_year']
df['milage_log'] = np.log1p(df['milage'])
df['miles_per_year'] = df['milage'] / (df['model_age'] + 1)

numeric_features     = ['model_year', 'model_age', 'milage_log', 'miles_per_year']
categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
high_card_features   = ['brand', 'model']

for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

# Zadržavamo originalnu cijenu unutar DataFrame-a privremeno radi usklađivanja indeksa
X = df[numeric_features + categorical_features + high_card_features + ['price']].copy()
y = np.log1p(df['price'])

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 💥 POPRAVAK INDEKSA: Izvlačimo dolarske vrijednosti preko .loc koristeći usklađene indekse
y_train_dollar = X_train['price'].values
X_train = X_train.drop(columns=['price'])
X_test = X_test.drop(columns=['price'])

# ── Target Encoding u DOLARSKOM prostoru ──────────────────────────────────────
KF          = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean_dollar = float(y_train_dollar.mean())
te_maps     = {'global_mean': global_mean_dollar}

for col in high_card_features:
    X_train[col + '_te'] = global_mean_dollar

    for fold_idx, (tr_idx, val_idx) in enumerate(KF.split(X_train)):
        fold_train_X = X_train.iloc[tr_idx]
        fold_train_y_dollar = y_train_dollar[tr_idx]

        fold_map = (pd.Series(fold_train_y_dollar, index=fold_train_X.index)
                      .groupby(fold_train_X[col].values)
                      .mean())

        X_train.loc[X_train.index[val_idx], col + '_te'] = (
            X_train.iloc[val_idx][col].map(fold_map).fillna(global_mean_dollar).values
        )

    # Mapiranje za testni skup radimo na temelju cijelog trening skupa
    full_map = pd.Series(y_train_dollar, index=X_train.index).groupby(X_train[col]).mean()
    te_maps[col] = full_map
    X_test[col + '_te'] = X_test[col].map(full_map).fillna(global_mean_dollar)

    X_train = X_train.drop(columns=[col])
    X_test  = X_test.drop(columns=[col])

te_rename = {col + '_te': col for col in high_card_features}
X_train = X_train.rename(columns=te_rename)
X_test  = X_test.rename(columns=te_rename)

# 💥 MAPE POPRAVAK 2: Težinski vektori (koristimo Target Encoded cijenu modela)
# Što je model automobila jeftiniji, to mu je veća težina kako bismo suzbili visoki MAPE
sample_weights = 1.0 / (X_train['model'].clip(lower=5000))
sample_weights = sample_weights / sample_weights.mean()  # Normalizacija težina

# ── ColumnTransformer ─────────────────────────────────────────────────────────
num_and_te_features = numeric_features + high_card_features

preprocessor = ColumnTransformer([
    ('num_and_te', StandardScaler(), num_and_te_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
     categorical_features)
], remainder='drop')

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

print(f"\nTraining matrix shape : {X_train_proc.shape}")
print(f"Test     matrix shape : {X_test_proc.shape}")

# ── Persist ───────────────────────────────────────────────────────────────────
np.save('data/X_train_proc.npy', X_train_proc)
np.save('data/X_test_proc.npy',  X_test_proc)
np.save('data/y_train.npy',      y_train.values)
np.save('data/y_test.npy',       y_test.values)
np.save('data/sample_weights.npy', sample_weights.values)

with open('data/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

with open('data/te_maps.pkl', 'wb') as f:
    pickle.dump(te_maps, f)