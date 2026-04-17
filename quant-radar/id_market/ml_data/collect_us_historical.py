"""
collect_us_historical.py
========================
Seed script — jalankan SEKALI secara manual untuk mengisi us_features.parquet
dengan data historis 3 tahun (2022–2024).

Setelah ini, append_daily.py yang handle penambahan harian via GH Actions.

Output: ml_data/us_features.parquet
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("ml_data")
OUTPUT_FILE = OUTPUT_DIR / "us_features.parquet"

START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"   # seed sampai akhir 2024; 2025+ via append harian

# Tickers yang di-fetch
TICKERS = {
    # Major indices
    "^GSPC":  "sp500",
    "^IXIC":  "nasdaq",
    "^DJI":   "dow",

    # Volatility & Dollar
    "^VIX":   "vix",
    "DX-Y.NYB": "dxy",

    # Commodities
    "GC=F":   "gold",
    "CL=F":   "crude",
    "MTF=F":  "coal",       # Coal futures (Rotterdam)

    # Sector ETFs (US)
    "XLF":    "xlf",        # Financials
    "XLK":    "xlk",        # Technology
    "XLE":    "xle",        # Energy
    "XLB":    "xlb",        # Materials
    "XLY":    "xly",        # Consumer Discretionary
    "XLI":    "xli",        # Industrials
    "XLP":    "xlp",        # Consumer Staples

    # Additional correlated assets
    "EEM":    "eem",        # Emerging Markets ETF (penting untuk IDX context)
    "ASHR":   "ashr",       # China A-shares (berpengaruh ke IDX)
    "^TNX":   "us10y",      # US 10Y yield
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_ticker_data(symbol: str, start: str, end: str) -> pd.Series | None:
    """Fetch closing price series untuk satu ticker."""
    try:
        df = yf.download(symbol, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            print(f"  [WARN] No data: {symbol}")
            return None
        # yfinance may return MultiIndex columns even for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                close = df["Close"].squeeze()
            else:
                close = df.xs("Close", axis=1, level=1).squeeze()
        else:
            close = df["Close"].squeeze()
        close.name = symbol
        return close
    except Exception as e:
        print(f"  [ERROR] {symbol}: {e}")
        return None


def build_features(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Dari raw close prices, hitung:
    - % change harian (pct_chg) untuk semua kecuali VIX & US10Y
    - Level absolute untuk VIX & US10Y (bukan pct_chg, karena yang relevan adalah level-nya)
    """
    features = pd.DataFrame(index=closes.index)

    for symbol, col_name in TICKERS.items():
        if symbol not in closes.columns:
            continue

        series = closes[symbol]

        if col_name in ("vix", "us10y"):
            # Level, bukan pct change
            features[f"{col_name}_level"] = series
            features[f"{col_name}_chg"]   = series.pct_change() * 100
        else:
            features[f"{col_name}_chg"] = series.pct_change() * 100

    # Drop row pertama (NaN dari pct_change)
    features = features.dropna(how="all")

    # Flag: apakah hari ini US market buka (ada data S&P500)
    features["us_market_open"] = features["sp500_chg"].notna().astype(int)

    # Tambahan derived features
    if "sp500_chg" in features and "nasdaq_chg" in features:
        # Risk appetite proxy: rata-rata sp500 + nasdaq
        features["risk_appetite"] = (
            features["sp500_chg"].fillna(0) + features["nasdaq_chg"].fillna(0)
        ) / 2

    if "sp500_chg" in features and "vix_chg" in features:
        # Fear index: VIX naik saat market turun, jadi ini biasanya negatif korelasi
        features["fear_greed_proxy"] = (
            features["sp500_chg"].fillna(0) - features["vix_chg"].fillna(0)
        )

    features.index.name = "date"
    return features


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"  Quant Radar — US Features Historical Collector")
    print(f"  Period : {START_DATE} → {END_DATE}")
    print(f"  Tickers: {len(TICKERS)}")
    print(f"{'='*60}\n")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Cek apakah file sudah ada
    if OUTPUT_FILE.exists():
        existing = pd.read_parquet(OUTPUT_FILE)
        print(f"[INFO] File sudah ada: {len(existing)} rows")
        print(f"[INFO] Last date: {existing.index.max().date()}")
        ans = input("\nOverwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("[ABORT] Cancelled.")
            return

    # Fetch semua tickers sekaligus (batch lebih efisien)
    print("[1/3] Fetching raw close prices...")
    symbols = list(TICKERS.keys())

    all_closes = {}
    for symbol in symbols:
        print(f"  → {symbol} ({TICKERS[symbol]})")
        s = fetch_ticker_data(symbol, START_DATE, END_DATE)
        if s is not None:
            all_closes[symbol] = s
        time.sleep(0.3)  # gentle rate limiting

    closes_df = pd.DataFrame(all_closes)
    closes_df.index = pd.to_datetime(closes_df.index)

    # Handle MultiIndex columns jika ada (yfinance kadang return MultiIndex)
    if isinstance(closes_df.columns, pd.MultiIndex):
        closes_df.columns = closes_df.columns.get_level_values(0)

    print(f"\n  Fetched {len(closes_df.columns)}/{len(symbols)} tickers")
    print(f"  Date range: {closes_df.index.min().date()} → {closes_df.index.max().date()}")
    print(f"  Total rows: {len(closes_df)}")

    # Build features
    print("\n[2/3] Building features...")
    features = build_features(closes_df)

    # Drop hari non-trading (semua NaN)
    before = len(features)
    features = features[features["us_market_open"] == 1]
    print(f"  Rows setelah filter trading days: {len(features)} (dropped {before - len(features)})")

    # Preview
    print("\n  Sample output (last 3 rows):")
    print(features.tail(3).to_string())
    print(f"\n  Columns ({len(features.columns)}): {list(features.columns)}")

    # Save
    print(f"\n[3/3] Saving to {OUTPUT_FILE}...")
    features.to_parquet(OUTPUT_FILE, engine="pyarrow", compression="snappy")
    print(f"  ✓ Saved: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    print(f"\n[DONE] {len(features)} trading days stored.")
    print(f"[NEXT] Jalankan collect_idx_historical.py untuk IDX OHLCV data.")


if __name__ == "__main__":
    main()
