import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import ensure_directories

def run_eda(df):
    print("\n--- Pokrećem Eksplorativnu Analizu Podataka (EDA) ---")
    ensure_directories()
    
    print(f"Veličina skupa podataka: {df.shape}")
    print(f"Nedostajuće vrijednosti:\n{df[['price', 'mileage', 'model_year']].isnull().sum()}")
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    prices = df['price'].dropna()
    
    sns.histplot(prices, bins=60, kde=True, color='crimson', ax=axes[0])
    axes[0].set_title('Price distribution (original)')
    axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    sns.histplot(np.log1p(prices), bins=60, kde=True, color='royalblue', ax=axes[1])
    axes[1].set_title('Price distribution (log-transformed)')
    
    plt.tight_layout()
    plt.savefig('results/01_price_distribution.png', dpi=150)
    plt.close()
    print("Spremljeno: results/01_price_distribution.png")