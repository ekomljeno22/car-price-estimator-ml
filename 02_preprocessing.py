from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import pickle
import os

os.makedirs('data', exist_ok=True)

df = pd.read_csv('used_cars.csv')

df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

df = df.dropna(subset=['price', 'milage', 'model_year'])

df['milage_log'] = np.log1p(df['milage'])

numeric_features = ['model_year', 'milage_log']
categorical_features = ['fuel_type', 'transmission', 'accident', 'clean_title']
high_card_features = ['brand', 'model']

for col in categorical_features:
    df[col] = df[col].fillna('Unknown')

X = df[numeric_features + categorical_features + high_card_features].copy()
y = np.log1p(df['price'])  # Log-transformation of target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

for col in high_card_features:
    train_temp = X_train.copy()
    train_temp['target_log'] = y_train
    
    te_map = train_temp.groupby(col)['target_log'].mean()
    
    global_mean = y_train.mean()
    X_train[col] = X_train[col].map(te_map).fillna(global_mean)
    X_test[col]  = X_test[col].map(te_map).fillna(global_mean)

num_and_te_features = numeric_features + high_card_features

preprocessor = ColumnTransformer([
    ('num_and_te', StandardScaler(), num_and_te_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
     categorical_features)
])

X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

print(f"Final shape of training matrix for neural network: {X_train_proc.shape}")
print(f"Final shape of test matrix for neural network: {X_test_proc.shape}")

np.save('data/X_train_proc.npy', X_train_proc)
np.save('data/X_test_proc.npy', X_test_proc)
np.save('data/y_train.npy', y_train.values)
np.save('data/y_test.npy', y_test.values)

# Save preprocessor for feature interpretation
with open('data/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

print("\nProcessed data saved successfully!")
