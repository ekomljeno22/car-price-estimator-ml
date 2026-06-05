import os
import json
import pandas as pd
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    with open(os.path.join(BASE_DIR, 'config.json'), 'r') as f:
        return json.load(f)

def load_and_clean_data(csv_path=None):
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, 'data', 'csv', 'used_cars.csv')
        
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Datoteka nije pronađena: {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    if 'milage' in df.columns:
        df = df.rename(columns={'milage': 'mileage'})
    
    df['price'] = pd.to_numeric(df['price'].astype(str).str.replace(r'[$,]', '', regex=True), errors='coerce')
    df['mileage'] = pd.to_numeric(df['mileage'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
    
    if 'engine' in df.columns:
        df['hp'] = df['engine'].astype(str).str.extract(r'(\d+\.?\d*)HP')[0].astype(float)
        df['liters'] = df['engine'].astype(str).str.extract(r'(\d+\.?\d*)L')[0].astype(float)
        
        df['hp'] = df['hp'].fillna(df['hp'].median())
        df['liters'] = df['liters'].fillna(df['liters'].median())
        
    return df

def ensure_directories():
    for folder in ['data', 'models', 'results']:
        os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)