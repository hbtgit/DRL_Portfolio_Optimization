import pandas as pd
import numpy as np
import os
import glob

# Constants
DATA_DIR = "data"
EQUITIES_DIR = os.path.join(DATA_DIR, "equities")
CRYPTO_DIR = os.path.join(DATA_DIR, "crypto")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

def calculate_indicators(df):
    """Calculates technical indicators for a given dataframe of a single asset."""
    # Ensure data is sorted by date
    df = df.sort_values('Date')
    
    # Simple Moving Averages
    df['sma_10'] = df['Close'].rolling(window=10).mean()
    df['sma_30'] = df['Close'].rolling(window=30).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['Close'].rolling(window=20).mean()
    df['bb_std'] = df['Close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
    df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
    
    # Log Returns
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df

def process_directory(directory, asset_type):
    all_data = []
    files = glob.glob(os.path.join(directory, "*.csv"))
    
    for file in files:
        ticker = os.path.basename(file).replace(".csv", "")
        print(f"Processing {ticker} ({asset_type})...")
        
        # The CSV has 3 header rows. We skip them and manually name the columns.
        cols = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df = pd.read_csv(file, skiprows=3, names=cols)
        
        # Standardize 'Date' column
        df['Date'] = pd.to_datetime(df['Date'])
        df['Ticker'] = ticker
        df['Asset_Type'] = asset_type
        
        df = calculate_indicators(df)
        all_data.append(df)
        
    return pd.concat(all_data, ignore_index=True)

def normalize_features(df, feature_cols):
    """Performs Z-score normalization on the specified columns."""
    df_norm = df.copy()
    for col in feature_cols:
        if col in df_norm.columns:
            mean = df_norm[col].mean()
            std = df_norm[col].std()
            if std != 0:
                df_norm[col] = (df_norm[col] - mean) / std
            else:
                df_norm[col] = 0
    return df_norm

if __name__ == "__main__":
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        print(f"Created directory: {PROCESSED_DIR}")

    # Process Equities and Crypto
    equities_df = process_directory(EQUITIES_DIR, "equity")
    crypto_df = process_directory(CRYPTO_DIR, "crypto")
    
    # Combine all assets
    full_df = pd.concat([equities_df, crypto_df], ignore_index=True)
    
    # Handle missing values created by indicators (e.g., first 30 days of SMA)
    # We forward fill then backward fill per ticker to ensure no NaNs remain
    print("Handling missing values...")
    full_df = full_df.sort_values(['Ticker', 'Date'])
    # Safer approach: fill NaNs per column within each ticker group
    cols_to_fix = [c for c in full_df.columns if c not in ['Ticker', 'Date', 'Asset_Type']]
    for col in cols_to_fix:
        full_df[col] = full_df.groupby('Ticker')[col].ffill().bfill()
    
    # Redo the combination to be safe
    raw_path = os.path.join(PROCESSED_DIR, "portfolio_data_raw.csv")
    full_df.to_csv(raw_path, index=False)
    print(f"Saved raw features to {raw_path}")
    
    # Normalize features (Z-score scaling per ticker for price-dependent features)
    # Note: RSI is already 0-100, but Z-scoring it per ticker is still fine.
    # Log returns are already roughly normalized, but Z-scoring helps.
    feature_columns = [
        'Close', 'Volume', 'sma_10', 'sma_30', 'rsi_14', 
        'macd', 'macd_signal', 'bb_middle', 'bb_std', 'bb_upper', 'bb_lower', 'log_return'
    ]
    
    print("Normalizing features per ticker...")
    # More robust approach using transform to preserve all columns naturally
    means = full_df.groupby('Ticker')[feature_columns].transform('mean')
    stds = full_df.groupby('Ticker')[feature_columns].transform('std')
    
    norm_df = full_df.copy()
    norm_df[feature_columns] = (norm_df[feature_columns] - means) / stds
    # Handle assets where std might be 0 (e.g., constant values)
    norm_df[feature_columns] = norm_df[feature_columns].fillna(0)
    
    # Final check: ensure Ticker is present
    if 'Ticker' not in norm_df.columns:
        print("Error: Ticker column lost. This should NOT happen with transform.")
    
    norm_path = os.path.join(PROCESSED_DIR, "portfolio_data_normalized.csv")
    norm_df.to_csv(norm_path, index=False)
    print(f"Saved normalized features to {norm_path}")
    
    print("\nFeature engineering complete.")
