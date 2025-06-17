import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def feature_engineering(df):
    df = df.copy()
    if 'VehicleYear' in df.columns:
        df['VehicleAge'] = 2025 - df['VehicleYear']
    if 'TotalPremium' in df.columns and 'TotalClaims' in df.columns:
        df['LossRatio'] = df['TotalClaims'] / df['TotalPremium'].replace(0, 1)
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    return df

def handle_missing_data(df):
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(include='object').columns
    
    df[num_cols] = SimpleImputer(strategy='mean').fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_cols])
    return df

def encode_categoricals(df, max_unique=20):
    cat_cols = [col for col in df.select_dtypes(include='object').columns if df[col].nunique() <= max_unique]
    return pd.get_dummies(df, columns=cat_cols, drop_first=True)

def prepare_data(df):
    df = feature_engineering(df)
    df = handle_missing_data(df)
    df = encode_categoricals(df)
    return df
