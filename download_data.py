import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# Assets
DOW_30_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "JNJ", "V",
    "WMT", "PG", "JPM", "MA", "NVDA", "HD", "CVX", "LLY", "ABBV", "PFE",
    "MRK", "PEP", "KO", "ORCL", "AVGO", "BAC", "COST", "TMO", "ACN", "CSCO"
]

CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

START_DATE = "2015-01-01"
END_DATE = "2023-12-31"

DATA_DIR = "data"

def download_data(tickers, start, end, subfolder):
    target_dir = os.path.join(DATA_DIR, subfolder)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        try:
            data = yf.download(ticker, start=start, end=end)
            if not data.empty:
                filename = f"{ticker.replace('-', '_')}.csv"
                filepath = os.path.join(target_dir, filename)
                data.to_csv(filepath)
                print(f"Saved {ticker} to {filepath}")
            else:
                print(f"No data found for {ticker}")
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created main data directory: {DATA_DIR}")

    print("--- Downloading Equities ---")
    download_data(DOW_30_TICKERS, START_DATE, END_DATE, "equities")

    print("\n--- Downloading Cryptocurrencies ---")
    download_data(CRYPTO_TICKERS, START_DATE, END_DATE, "crypto")

    print("\nData acquisition complete.")
