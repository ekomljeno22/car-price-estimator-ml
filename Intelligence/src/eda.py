import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import ensure_directories

def run_eda(df):
    print("\n--- Pokrećem Eksplorativnu Analizu Podataka (EDA) ---")
    ensure_directories()
    
    mileage_col = 'mileage' if 'mileage' in df.columns else 'milage'
    
    print(f"Veličina skupa podataka: {df.shape}")
    available_cols = [col for col in ['price', mileage_col, 'model_year'] if col in df.columns]
    print(f"Nedostajuće vrijednosti:\n{df[available_cols].isnull().sum()}")
    print(f"\nStatistički pregled:\n{df.describe()}")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    prices = df['price'].dropna()
    
    sns.histplot(prices, bins=60, kde=True, color='crimson', ax=axes[0])
    axes[0].set_title('Price distribution (original)')
    axes[0].set_xlabel('Price (USD)')
    axes[0].set_ylabel('Frequency')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    sns.histplot(np.log1p(prices), bins=60, kde=True, color='royalblue', ax=axes[1])
    axes[1].set_title('Price distribution (log-transformed)')
    axes[1].set_xlabel('log(Price + 1)')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/01_price_distribution.png', dpi=150)
    plt.close()
    print("Spremljeno: results/01_price_distribution.png")

    def iqr_bounds(series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - 1.5 * iqr, q3 + 1.5 * iqr

    cols_to_check = [col for col in ['price', mileage_col] if col in df.columns]
    
    for col in cols_to_check:
        s = df[col].dropna()
        if len(s) > 0:
            lo, hi = iqr_bounds(s)
            n_out = ((s < lo) | (s > hi)).sum()
            pct = 100 * n_out / len(s)
            print(f"{col} IQR bounds: [{lo:,.0f}, {hi:,.0f}]  →  {n_out} outliers ({pct:.1f}%)")

    fig, axes = plt.subplots(1, len(cols_to_check), figsize=(13, 5))
    if len(cols_to_check) == 1:
        axes = [axes]
        
    for ax, col, color in zip(axes, cols_to_check, ['crimson', 'steelblue']):
        sns.boxplot(x=df[col].dropna(), color=color, ax=ax, fliersize=2)
        ax.set_title(f'{col} – boxplot')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        
    plt.tight_layout()
    plt.savefig('results/02_outlier_boxplots.png', dpi=150)
    plt.close()
    print("Spremljeno: results/02_outlier_boxplots.png")

    if 'price' in df.columns and mileage_col in df.columns:
        sample = df[['price', mileage_col]].dropna().sample(min(5000, len(df)), random_state=42)
        
        plt.figure(figsize=(9, 6))
        plt.scatter(sample[mileage_col], sample['price'], alpha=0.3, s=15, color='teal', edgecolors='none')
        plt.xlabel('Mileage (mi)')
        plt.ylabel('Price (USD)')
        plt.title(f'Mileage vs Price (sample n={len(sample):,})')
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        plt.tight_layout()
        plt.savefig('results/03_mileage_vs_price.png', dpi=150)
        plt.close()
        print("Spremljeno: results/03_mileage_vs_price.png")

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
        print("Spremljeno: results/04_correlation_heatmap.png")

    if 'brand' in df.columns and 'price' in df.columns:
        brand_stats = (df.groupby('brand')['price']
                       .agg(median_price='median', count='count')
                       .query('count >= 30')
                       .sort_values('median_price', ascending=False)
                       .head(20))

        if not brand_stats.empty:
            plt.figure(figsize=(12, 6))
            plt.bar(brand_stats.index, brand_stats['median_price'],
                    color=plt.cm.viridis(np.linspace(0.2, 0.85, len(brand_stats))))
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('Median Price (USD)')
            plt.title('Top 20 Brands – Median Price (min 30 listings)')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            plt.tight_layout()
            plt.savefig('results/05_brand_median_price.png', dpi=150)
            plt.close()
            print("Spremljeno: results/05_brand_median_price.png")

    if 'model_year' in df.columns and 'price' in df.columns:
        year_stats = df.groupby('model_year')['price'].median().reset_index()
        
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=year_stats, x='model_year', y='price', marker='o', color='darkorange', lw=2.5)
        plt.xlabel('Model Year')
        plt.ylabel('Median Price (USD)')
        plt.title('Median Price Trend by Model Year')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
        plt.tight_layout()
        plt.savefig('results/06_price_by_year.png', dpi=150)
        plt.close()
        print("Spremljeno: results/06_price_by_year.png")

    if 'fuel_type' in df.columns:
        fuel_counts = df['fuel_type'].value_counts().head(10)
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=fuel_counts.values, y=fuel_counts.index, hue=fuel_counts.index, legend=False, palette='viridis')
        plt.xlabel('Count')
        plt.ylabel('Fuel Type')
        plt.title('Top Fuel Types Distribution')
        plt.tight_layout()
        plt.savefig('results/07_fuel_type_distribution.png', dpi=150)
        plt.close()
        print("Spremljeno: results/07_fuel_type_distribution.png")

    if 'accident' in df.columns and 'price' in df.columns:
        accident_df = df[['accident', 'price']].dropna()
        
        if not accident_df.empty:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=accident_df, x='accident', y='price', hue='accident', legend=False, palette='Set2', showfliers=False)
            plt.xlabel('Accident Reported')
            plt.ylabel('Price (USD) - Outliers Hidden')
            plt.title('Price Distribution: Accident vs Clean Title')
            plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
            plt.tight_layout()
            plt.savefig('results/08_accident_vs_price.png', dpi=150)
            plt.close()
            print("Spremljeno: results/08_accident_vs_price.png")