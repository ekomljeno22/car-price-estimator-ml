import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('results', exist_ok=True)

# ── Load & clean ──────────────────────────────────────────────────────────────
df = pd.read_csv('used_cars.csv')

df['price']  = pd.to_numeric(df['price'].str.replace(r'[$,]',  '', regex=True), errors='coerce')
df['milage'] = pd.to_numeric(df['milage'].str.replace(r'[^\d.]', '', regex=True), errors='coerce')

print(f"Dataset size (rows, columns): {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistical summary:\n{df.describe()}")

# ── 1. Price distributions ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(df['price'].dropna(), bins=60, kde=True, color='crimson', ax=axes[0])
axes[0].set_title('Price distribution (original)')
axes[0].set_xlabel('Price (USD)')
axes[0].set_ylabel('Frequency')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

sns.histplot(np.log1p(df['price'].dropna()), bins=60, kde=True, color='royalblue', ax=axes[1])
axes[1].set_title('Price distribution (log-transformed)')
axes[1].set_xlabel('log(Price + 1)')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('results/01_price_distribution.png', dpi=150)
plt.close()
print("Saved: results/01_price_distribution.png")

# ── 2. Outlier analysis (IQR) ─────────────────────────────────────────────────
def iqr_bounds(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr

for col in ['price', 'milage']:
    s = df[col].dropna()
    lo, hi = iqr_bounds(s)
    n_out = ((s < lo) | (s > hi)).sum()
    pct   = 100 * n_out / len(s)
    print(f"\n{col} IQR bounds: [{lo:,.0f}, {hi:,.0f}]  →  {n_out} outliers ({pct:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, col, color in zip(axes, ['price', 'milage'], ['crimson', 'steelblue']):
    sns.boxplot(x=df[col].dropna(), color=color, ax=ax, fliersize=2)
    ax.set_title(f'{col} – boxplot')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.tight_layout()
plt.savefig('results/02_outlier_boxplots.png', dpi=150)
plt.close()
print("Saved: results/02_outlier_boxplots.png")

# ── 3. Mileage vs Price scatter ───────────────────────────────────────────────
sample = df[['price', 'milage']].dropna().sample(min(5000, len(df)), random_state=42)

plt.figure(figsize=(9, 6))
plt.scatter(sample['milage'], sample['price'], alpha=0.3, s=15, color='teal', edgecolors='none')
plt.xlabel('Mileage (mi)')
plt.ylabel('Price (USD)')
plt.title('Mileage vs Price (sample n=5 000)')
plt.xaxis = plt.gca().xaxis
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
plt.tight_layout()
plt.savefig('results/03_milage_vs_price.png', dpi=150)
plt.close()
print("Saved: results/03_milage_vs_price.png")

# ── 4. Correlation heatmap (numeric) ─────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(max(6, len(numeric_cols)), max(5, len(numeric_cols) - 1)))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation matrix – numeric features')
    plt.tight_layout()
    plt.savefig('results/04_correlation_heatmap.png', dpi=150)
    plt.close()
    print("Saved: results/04_correlation_heatmap.png")

# ── 5. Top-20 brands by median price ─────────────────────────────────────────
if 'brand' in df.columns:
    brand_stats = (df.groupby('brand')['price']
                     .agg(median_price='median', count='count')
                     .query('count >= 30')
                     .sort_values('median_price', ascending=False)
                     .head(20))

    plt.figure(figsize=(12, 6))
    bars = plt.bar(brand_stats.index, brand_stats['median_price'],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, len(brand_stats))))
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Median Price (USD)')
    plt.title('Top 20 Brands – Median Price (min 30 listings)')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.tight_layout()
    plt.savefig('results/05_brand_median_price.png', dpi=150)
    plt.close()
    print("Saved: results/05_brand_median_price.png")