import pandas as pd
import numpy as np
from pathlib import Path

# ========================
# CONFIG
# ========================

BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "master.parquet"
FEATURE_FILE = BASE_DIR / "features.parquet"

# ========================
# SANITY CHECK
# ========================

print("\n=== FEATURE BUILDER ===")

if not MASTER_FILE.exists():
    raise Exception("master.parquet tidak ditemukan. Jalankan cleaning_pipeline dulu.")

print("Loading master dataset...")
df = pd.read_parquet(MASTER_FILE)

print("Rows:", len(df))

# ========================
# SORT (WAJIB untuk rolling)
# ========================

df = df.sort_values(["ticker", "date"])

# ========================
# FEATURE ENGINEERING
# ========================

print("Building features...")

# ---------- NET FOREIGN FLOW ----------
if "foreign_buy" in df.columns and "foreign_sell" in df.columns:
    df["foreign_net"] = df["foreign_buy"] - df["foreign_sell"]

# ---------- VALUE PER TRADE ----------
if "value" in df.columns and "freq" in df.columns:
    df["value_per_trade"] = df["value"] / df["freq"].replace(0, np.nan)

# ---------- PRICE CHANGE ----------
if "close" in df.columns:
    df["return_1d"] = df.groupby("ticker")["close"].pct_change()

# ---------- VOLUME SPIKE ----------
if "volume" in df.columns:
    df["volume_ma20"] = (
        df.groupby("ticker")["volume"]
        .transform(lambda x: x.rolling(20).mean())
    )
    df["volume_spike"] = df["volume"] / df["volume_ma20"]

# ---------- FREQ SPIKE ----------
if "freq" in df.columns:
    df["freq_ma20"] = (
        df.groupby("ticker")["freq"]
        .transform(lambda x: x.rolling(20).mean())
    )
    df["freq_spike"] = df["freq"] / df["freq_ma20"]

# ---------- BID / OFFER PRESSURE ----------
if "bid_volume" in df.columns and "offer_volume" in df.columns:
    total = df["bid_volume"] + df["offer_volume"]
    df["bid_pressure"] = df["bid_volume"] / total.replace(0, np.nan)

# ========================
# CLEAN RESULT
# ========================

df = df.replace([np.inf, -np.inf], np.nan)

# ========================
# VALIDATION
# ========================

print("\n=== FEATURE SUMMARY ===")
print("Columns:", len(df.columns))
print("Feature columns added:", [
    c for c in df.columns
    if c not in [
        "ticker","date","open","high","low","close",
        "volume","value","freq","foreign_buy","foreign_sell",
        "offer_volume","bid_volume",
        "non_regular_volume","non_regular_value","non_regular_freq"
    ]
])

print("Missing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

# ========================
# SAVE FEATURE STORE
# ========================

print("\nSaving features...")
df.to_parquet(FEATURE_FILE, index=False)

print("\n✅ DONE")
print("Saved:", FEATURE_FILE)
print("Rows:", len(df))
