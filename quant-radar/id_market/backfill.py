"""
backfill.py
===========
Extend us_features.parquet dan idx_ohlcv.parquet
dari 2025-01-01 sampai 2026-04-16 (termasuk).

Setelah ini:
  - jalankan train.py   (retrain dengan data lengkap)
  - jalankan inference.py (prediksi IDX 17 April 2026)

Usage: python backfill.py
"""

import warnings
import time
from pathlib import Path

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("ml_data")
US_FILE    = DATA_DIR / "us_features.parquet"
IDX_FILE   = DATA_DIR / "idx_ohlcv.parquet"

BACKFILL_START = "2025-01-01"
BACKFILL_END   = "2026-04-16"   # inclusive, ambil sampai 16 April
BATCH_SIZE     = 50
SLEEP_BATCH    = 1.5

US_TICKERS = {
    "^GSPC":    "sp500",
    "^IXIC":    "nasdaq",
    "^DJI":     "dow",
    "^VIX":     "vix",
    "DX-Y.NYB": "dxy",
    "GC=F":     "gold",
    "CL=F":     "crude",
    "MTF=F":    "coal",
    "XLF":      "xlf",
    "XLK":      "xlk",
    "XLE":      "xle",
    "XLB":      "xlb",
    "XLY":      "xly",
    "XLI":      "xli",
    "XLP":      "xlp",
    "EEM":      "eem",
    "ASHR":     "ashr",
    "^TNX":     "us10y",
}


# ── US Backfill ───────────────────────────────────────────────────────────────
def fetch_us_close(symbol: str, start: str, end: str) -> pd.Series | None:
    try:
        df = yf.download(symbol, start=start, end=end,
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
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


def build_us_features(closes: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=closes.index)
    for symbol, col_name in US_TICKERS.items():
        if symbol not in closes.columns:
            continue
        series = closes[symbol]
        if col_name in ("vix", "us10y"):
            features[f"{col_name}_level"] = series
            features[f"{col_name}_chg"]   = series.pct_change() * 100
        else:
            features[f"{col_name}_chg"] = series.pct_change() * 100
    features = features.dropna(how="all")
    features["us_market_open"] = features["sp500_chg"].notna().astype(int)
    if "sp500_chg" in features and "nasdaq_chg" in features:
        features["risk_appetite"] = (
            features["sp500_chg"].fillna(0) + features["nasdaq_chg"].fillna(0)
        ) / 2
    if "sp500_chg" in features and "vix_chg" in features:
        features["fear_greed_proxy"] = (
            features["sp500_chg"].fillna(0) - features["vix_chg"].fillna(0)
        )
    features.index.name = "date"
    return features


def backfill_us():
    print("── US Features Backfill ─────────────────────────────")

    # Load existing
    existing = pd.read_parquet(US_FILE)
    existing.index = pd.to_datetime(existing.index)
    last_date = existing.index.max()
    print(f"  Existing last date : {last_date.date()}")
    print(f"  Backfill period    : {BACKFILL_START} → {BACKFILL_END}")

    # Fetch
    all_closes = {}
    for symbol in US_TICKERS:
        print(f"  → {symbol}")
        s = fetch_us_close(symbol, BACKFILL_START, BACKFILL_END)
        if s is not None:
            all_closes[symbol] = s
        time.sleep(0.2)

    closes_df = pd.DataFrame(all_closes)
    closes_df.index = pd.to_datetime(closes_df.index)
    if hasattr(closes_df.index, 'tz') and closes_df.index.tz is not None:
        closes_df.index = closes_df.index.tz_localize(None)

    new_features = build_us_features(closes_df)
    new_features = new_features[new_features["us_market_open"] == 1]

    # Drop overlap dengan existing
    new_only = new_features[new_features.index > last_date]
    print(f"  New rows to append : {len(new_only)}")

    if new_only.empty:
        print("  [SKIP] Tidak ada data baru.")
        return

    # Concat & save
    combined = pd.concat([existing, new_only])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    combined.to_parquet(US_FILE, engine="pyarrow", compression="snappy")
    print(f"  ✓ US parquet updated: {len(combined)} rows total")
    print(f"  New last date: {combined.index.max().date()}")


# ── IDX Backfill ──────────────────────────────────────────────────────────────
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    def per_ticker(g):
        g = g.copy()
        c = g["close"]
        v = g["volume"]
        g["ret_1d"]       = c.pct_change(1) * 100
        g["ret_3d"]       = c.pct_change(3) * 100
        g["ret_5d"]       = c.pct_change(5) * 100
        g["ret_20d"]      = c.pct_change(20) * 100
        g["vol_20d_avg"]  = v.rolling(20).mean()
        g["vol_ratio"]    = v / g["vol_20d_avg"]
        g["volatility_20d"] = (c.pct_change() * 100).rolling(20).std()
        g["ma5"]          = c.rolling(5).mean()
        g["ma20"]         = c.rolling(20).mean()
        g["above_ma5"]    = (c > g["ma5"]).astype(int)
        g["above_ma20"]   = (c > g["ma20"]).astype(int)
        g["next_day_ret"] = c.pct_change(1).shift(-1) * 100
        g["label"]        = (g["next_day_ret"] > 3.0).astype(int)
        return g

    return df.groupby("ticker", group_keys=False).apply(per_ticker)


def fetch_idx_batch(tickers: list, start: str, end: str) -> pd.DataFrame:
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False, group_by="ticker")
    except Exception as e:
        print(f"    [ERROR] {e}")
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    records = []
    lvl0 = raw.columns.get_level_values(0).unique().tolist()
    field_first = any(x in lvl0 for x in ["Close", "Open", "High", "Low", "Volume"])

    if len(tickers) == 1:
        ticker = tickers[0]
        df = raw.copy()
        if isinstance(df.columns, pd.MultiIndex):
            if field_first:
                df = pd.DataFrame({
                    "open": raw["Open"].squeeze() if "Open" in raw.columns.get_level_values(0) else np.nan,
                    "high": raw["High"].squeeze() if "High" in raw.columns.get_level_values(0) else np.nan,
                    "low":  raw["Low"].squeeze()  if "Low"  in raw.columns.get_level_values(0) else np.nan,
                    "close":raw["Close"].squeeze(),
                    "volume":raw["Volume"].squeeze() if "Volume" in raw.columns.get_level_values(0) else np.nan,
                }, index=raw.index)
            else:
                df = raw[ticker].copy()
                df.columns = [c.lower() for c in df.columns]
                df = df.drop(columns=["adj close"], errors="ignore")
        else:
            df.columns = [c.lower() for c in df.columns]
            df = df.drop(columns=["adj close"], errors="ignore")

        df = df.dropna(subset=["close"])
        if not df.empty:
            df["ticker"] = ticker.replace(".JK", "")
            df.index.name = "date"
            df = df.reset_index()
            records.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
    else:
        for ticker in tickers:
            try:
                if field_first:
                    if ticker not in raw.columns.get_level_values(1):
                        continue
                    df = pd.DataFrame({
                        "open":   raw["Open"][ticker]   if "Open"   in raw.columns.get_level_values(0) else np.nan,
                        "high":   raw["High"][ticker]   if "High"   in raw.columns.get_level_values(0) else np.nan,
                        "low":    raw["Low"][ticker]    if "Low"    in raw.columns.get_level_values(0) else np.nan,
                        "close":  raw["Close"][ticker],
                        "volume": raw["Volume"][ticker] if "Volume" in raw.columns.get_level_values(0) else np.nan,
                    }, index=raw.index)
                else:
                    if ticker not in raw.columns.get_level_values(0):
                        continue
                    sub = raw[ticker].copy()
                    sub.columns = [c.lower() for c in sub.columns]
                    sub = sub.drop(columns=["adj close"], errors="ignore")
                    df = sub[["open", "high", "low", "close", "volume"]]

                df = df.dropna(subset=["close"])
                if df.empty:
                    continue
                df = df.copy()
                df["ticker"] = ticker.replace(".JK", "")
                df.index.name = "date"
                df = df.reset_index()
                records.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
            except Exception:
                continue

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def backfill_idx():
    print("\n── IDX OHLCV Backfill ───────────────────────────────")

    existing = pd.read_parquet(IDX_FILE)
    existing["date"] = pd.to_datetime(existing["date"])

    # Last date per ticker
    last_dates = existing.groupby("ticker")["date"].max()
    overall_last = last_dates.max()
    print(f"  Existing last date : {overall_last.date()}")
    print(f"  Unique tickers     : {existing['ticker'].nunique()}")

    tickers = existing["ticker"].unique().tolist()
    tickers_jk = [f"{t}.JK" for t in tickers]

    batches = [tickers_jk[i:i+BATCH_SIZE] for i in range(0, len(tickers_jk), BATCH_SIZE)]
    print(f"  Fetching {len(tickers)} tickers in {len(batches)} batches...")

    all_new = []
    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i:2d}/{len(batches)} — {batch[0].replace('.JK','')} … {batch[-1].replace('.JK','')}")
        df_batch = fetch_idx_batch(batch, BACKFILL_START, BACKFILL_END)
        if not df_batch.empty:
            all_new.append(df_batch)
            print(f"    ✓ {df_batch['ticker'].nunique()} tickers, {len(df_batch)} rows")
        if i < len(batches):
            time.sleep(SLEEP_BATCH)

    if not all_new:
        print("  [ERROR] No new IDX data fetched.")
        return

    df_new = pd.concat(all_new, ignore_index=True)
    df_new["date"] = pd.to_datetime(df_new["date"])
    if hasattr(df_new["date"].dt, "tz") and df_new["date"].dt.tz is not None:
        df_new["date"] = df_new["date"].dt.tz_localize(None)

    # Only keep rows newer than existing per ticker
    df_new = df_new.merge(
        last_dates.reset_index().rename(columns={"date": "last_date"}),
        on="ticker", how="left"
    )
    df_new = df_new[df_new["date"] > df_new["last_date"]]
    df_new = df_new.drop(columns=["last_date"])

    print(f"\n  New rows to append : {len(df_new):,}")
    if df_new.empty:
        print("  [SKIP] Tidak ada data baru.")
        return

    # Recompute derived features on combined data (rolling windows need history)
    print("  Recomputing derived features on combined dataset...")

    # Combine existing raw OHLCV + new
    existing_raw = existing[["date", "ticker", "open", "high", "low", "close", "volume"]]
    combined_raw = pd.concat([existing_raw, df_new], ignore_index=True)
    combined_raw = combined_raw.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined_raw = combined_raw.sort_values(["ticker", "date"]).reset_index(drop=True)

    combined_featured = add_derived_features(combined_raw)
    combined_featured = combined_featured.dropna(subset=["ret_5d", "volatility_20d"])

    combined_featured.to_parquet(IDX_FILE, engine="pyarrow",
                                 compression="snappy", index=False)
    print(f"  ✓ IDX parquet updated: {len(combined_featured):,} rows")
    print(f"  New last date: {combined_featured['date'].max().date()}")
    print(f"  Unique tickers: {combined_featured['ticker'].nunique()}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"  Quant Radar — Backfill 2025-01-01 → 2026-04-16")
    print(f"{'='*60}\n")

    backfill_us()
    backfill_idx()

    print(f"\n{'='*60}")
    print(f"  BACKFILL DONE")
    print(f"  Next: python train.py → python inference.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()