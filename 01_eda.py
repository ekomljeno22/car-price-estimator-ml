import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

df = pd.read_csv('used_cars.csv')

df['price'] = df['price'].str.replace('$', '').str.replace(',', '')
df['price'] = pd.to_numeric(df['price'], errors='coerce')

df['milage'] = df['milage'].str.replace(' mi.', '').str.replace(',', '')
df['milage'] = pd.to_numeric(df['milage'], errors='coerce')

print(f"Dataset size (rows, columns): {df.shape}")
print(f"\nData types by columns:\n{df.dtypes}")
print(f"\nNumber of missing values by columns:\n{df.isnull().sum()}")
print(f"\nStatistical summary of numerical features:\n{df.describe()}")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df['price'], bins=50, kde=True, color='crimson')
plt.title('Price distribution (original)')
plt.xlabel('Price (USD)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(np.log1p(df['price']), bins=50, kde=True, color='royalblue')
plt.title('Price distribution (log-transformation)')
plt.xlabel('log(Price + 1)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/distribucija_cijena.png', dpi=150)
plt.show()
