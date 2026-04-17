"""
inference.py
=============
Daily inference — jalankan setelah US market close (jam 4 pagi WIB).
Membaca US features hari ini, lalu prediksi IDX ticker mana yang
berpotensi naik >3% di sesi pagi IDX (jam 9 WIB).

Input  : ml_data/model.json, ml_data/feature_cols.json,
         ml_data/us_features.parquet, ml_data/idx_ohlcv.parquet
Output : ml_data/inference_latest.parquet  (full results)
         ml_data/inference_latest.json     (top picks, untuk dashboard)
"""

import json
import warnings
from datetime import datetime, timezone, date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("ml_data")
MODEL_FILE    = DATA_DIR / "model.json"
FEAT_FILE     = DATA_DIR / "feature_cols.json"
US_FILE       = DATA_DIR / "us_features.parquet"
IDX_FILE      = DATA_DIR / "idx_ohlcv.parquet"
OUT_PARQUET   = DATA_DIR / "inference_latest.parquet"
OUT_JSON      = DATA_DIR / "inference_latest.json"

TOP_N             = 20     # top picks yang ditampilkan di output
INFERENCE_THRESH  = 0.55   # sedikit diturunkan dari 0.6 biar recall lebih baik
MIN_VOLATILITY    = 0.5    # filter ticker yang terlalu "flat" (deadstock)
MAX_VOLATILITY    = 15.0   # filter ticker yang terlalu liar (pump & dump prone)

# US feature columns (harus sama persis dengan train.py)
US_FEATURE_COLS = [
    "sp500_chg", "nasdaq_chg", "dow_chg",
    "vix_level", "vix_chg",
    "dxy_chg", "gold_chg", "crude_chg", "coal_chg",
    "xlf_chg", "xlk_chg", "xle_chg", "xlb_chg",
    "xly_chg", "xli_chg", "xlp_chg",
    "eem_chg", "ashr_chg",
    "us10y_level", "us10y_chg",
    "risk_appetite", "fear_greed_proxy",
]

IDX_FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_20d",
    "vol_ratio", "volatility_20d",
    "above_ma5", "above_ma20",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_latest_us_features() -> tuple[pd.Series, date]:
    """
    Ambil US features dari hari trading terakhir yang tersedia.
    Ini adalah US close yang sudah settled — siap dipakai untuk prediksi IDX hari ini.
    """
    us = pd.read_parquet(US_FILE)
    us.index = pd.to_datetime(us.index)
    us = us.sort_index()

    # Ambil row terakhir (US trading day terakhir)
    latest_row  = us.iloc[-1]
    latest_date = us.index[-1].date()

    # Filter hanya kolom yang dibutuhkan
    available = [c for c in US_FEATURE_COLS if c in latest_row.index]
    missing   = [c for c in US_FEATURE_COLS if c not in latest_row.index]

    if missing:
        print(f"  [WARN] Missing US features: {missing}")

    return latest_row[available], latest_date


def get_latest_idx_features() -> pd.DataFrame:
    """
    Ambil ticker-level features dari IDX data.
    Untuk setiap ticker, ambil row terakhir (most recent state).
    Ini merepresentasikan kondisi ticker SEBELUM sesi hari ini.
    """
    idx = pd.read_parquet(IDX_FILE)
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values(["ticker", "date"])

    # Ambil row terakhir per ticker
    latest_per_ticker = idx.groupby("ticker").last().reset_index()

    # Filter IDX feature cols yang tersedia
    available_cols = ["ticker", "date", "close"] + [
        c for c in IDX_FEATURE_COLS if c in latest_per_ticker.columns
    ]
    return latest_per_ticker[available_cols]


def build_inference_input(
    us_features: pd.Series,
    idx_features: pd.DataFrame,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Build inference matrix:
    - Setiap row = 1 ticker
    - Broadcast US features (sama untuk semua ticker)
    - Join dengan IDX ticker-level features
    """
    n = len(idx_features)

    # Broadcast US features ke semua tickers
    for col in US_FEATURE_COLS:
        if col in us_features.index:
            idx_features = idx_features.copy()
            idx_features[col] = us_features[col]
        else:
            idx_features[col] = np.nan

    # Ensure semua feature_cols ada
    for col in feature_cols:
        if col not in idx_features.columns:
            idx_features[col] = np.nan

    X = idx_features[feature_cols].fillna(idx_features[feature_cols].median())
    return X, idx_features


def run_inference(
    model: xgb.XGBClassifier,
    X: pd.DataFrame,
    idx_features: pd.DataFrame,
    us_date: date,
) -> pd.DataFrame:
    """
    Run inference dan return sorted results DataFrame.
    """
    proba = model.predict_proba(X)[:, 1]

    results = idx_features[["ticker", "date", "close"]].copy()
    results["us_signal_date"] = us_date
    results["proba_up"]       = proba
    results["signal"]         = (proba >= INFERENCE_THRESH).astype(int)

    # Tambah IDX features ke output untuk context
    for col in IDX_FEATURE_COLS:
        if col in idx_features.columns:
            results[col] = idx_features[col].values

    # Filter volatility extremes
    if "volatility_20d" in results.columns:
        before = len(results)
        results = results[
            (results["volatility_20d"] >= MIN_VOLATILITY) &
            (results["volatility_20d"] <= MAX_VOLATILITY)
        ]
        filtered = before - len(results)
        if filtered > 0:
            print(f"  Filtered {filtered} tickers (volatility out of range)")

    # Sort by probability descending
    results = results.sort_values("proba_up", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    return results


def classify_signal_type(row: pd.Series) -> str:
    """
    Klasifikasi sinyal berdasarkan karakteristik ticker.
    Mirip dengan Quant Radar US logic.
    """
    vol_ratio    = row.get("vol_ratio", 1.0)
    ret_1d       = row.get("ret_1d", 0.0)
    ret_5d       = row.get("ret_5d", 0.0)
    volatility   = row.get("volatility_20d", 1.0)

    if vol_ratio > 2.0 and ret_1d > 1.0:
        return "BURST"
    elif ret_5d > 3.0 and volatility < 3.0:
        return "COMPOUNDER"
    elif ret_1d > 0 and ret_5d > 0 and volatility < 4.0:
        return "STEADY"
    else:
        return "SPECULATIVE"


def format_output(results: pd.DataFrame, top_n: int, us_date: date) -> dict:
    """
    Format top picks ke JSON-serializable dict untuk dashboard.
    """
    top = results[results["signal"] == 1].head(top_n)

    # Kalau yang di atas threshold kurang dari top_n, ambil top_n by proba
    if len(top) < top_n:
        print(f"  [INFO] Only {len(top)} tickers above threshold {INFERENCE_THRESH}, "
              f"padding with top-{top_n} by probability")
        top = results.head(top_n)

    picks = []
    for _, row in top.iterrows():
        signal_type = classify_signal_type(row)
        picks.append({
            "rank":           int(row["rank"]),
            "ticker":         row["ticker"],
            "proba_up":       round(float(row["proba_up"]), 4),
            "signal":         int(row["signal"]),
            "signal_type":    signal_type,
            "last_close":     round(float(row["close"]), 0),
            "ret_1d":         round(float(row.get("ret_1d", 0)), 2),
            "ret_5d":         round(float(row.get("ret_5d", 0)), 2),
            "vol_ratio":      round(float(row.get("vol_ratio", 1)), 2),
            "volatility_20d": round(float(row.get("volatility_20d", 0)), 2),
            "above_ma5":      int(row.get("above_ma5", 0)),
            "above_ma20":     int(row.get("above_ma20", 0)),
        })

    # US market context
    us_context = {}
    us_pq = pd.read_parquet(US_FILE)
    us_pq.index = pd.to_datetime(us_pq.index)
    latest_us = us_pq.iloc[-1]
    for col in ["sp500_chg", "nasdaq_chg", "dow_chg", "vix_level",
                "eem_chg", "risk_appetite", "fear_greed_proxy"]:
        if col in latest_us.index:
            us_context[col] = round(float(latest_us[col]), 4)

    # Summary stats
    above_thresh = int((results["signal"] == 1).sum())
    avg_proba    = float(results["proba_up"].mean())

    output = {
        "generated_at":       datetime.now(timezone.utc).isoformat(),
        "us_signal_date":     str(us_date),
        "idx_session":        "next open (09:00 WIB)",
        "inference_threshold": INFERENCE_THRESH,
        "total_tickers_scored": len(results),
        "tickers_above_threshold": above_thresh,
        "avg_proba_universe": round(avg_proba, 4),
        "us_market_context":  us_context,
        "top_picks":          picks,
    }

    return output


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"  Quant Radar — IDX Daily Inference")
    print(f"  Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Load model & feature list
    print("[1/5] Loading model...")
    if not MODEL_FILE.exists():
        print(f"  [ERROR] Model not found: {MODEL_FILE}")
        print(f"  Jalankan train.py terlebih dahulu.")
        return

    model = xgb.XGBClassifier()
    model.load_model(MODEL_FILE)

    with open(FEAT_FILE) as f:
        feature_cols = json.load(f)
    print(f"  ✓ Model loaded, {len(feature_cols)} features")

    # Load US features (latest trading day)
    print("\n[2/5] Loading US features...")
    us_features, us_date = get_latest_us_features()
    print(f"  ✓ US signal date : {us_date}")
    print(f"  S&P500 chg       : {us_features.get('sp500_chg', float('nan')):.2f}%")
    print(f"  Nasdaq chg       : {us_features.get('nasdaq_chg', float('nan')):.2f}%")
    print(f"  VIX level        : {us_features.get('vix_level', float('nan')):.2f}")
    print(f"  Risk appetite    : {us_features.get('risk_appetite', float('nan')):.2f}")

    # Load IDX ticker features
    print("\n[3/5] Loading IDX ticker features...")
    idx_features = get_latest_idx_features()
    print(f"  ✓ Tickers loaded : {len(idx_features)}")

    # Build inference matrix
    print("\n[4/5] Running inference...")
    X, idx_with_us = build_inference_input(us_features, idx_features, feature_cols)
    results = run_inference(model, X, idx_with_us, us_date)

    above  = (results["signal"] == 1).sum()
    print(f"  ✓ Scored {len(results)} tickers")
    print(f"  Above threshold ({INFERENCE_THRESH}): {above} tickers")

    # Top picks preview
    print(f"\n  ── Top 10 Picks ──────────────────────────────")
    top10 = results.head(10)
    for _, row in top10.iterrows():
        stype = classify_signal_type(row)
        bar   = "▓" * int(row["proba_up"] * 20)
        print(f"  {int(row['rank']):3d}. {row['ticker']:<6}  "
              f"proba={row['proba_up']:.3f}  {bar:<12}  "
              f"ret1d={row.get('ret_1d', 0):+.1f}%  "
              f"volR={row.get('vol_ratio', 1):.1f}x  "
              f"[{stype}]")

    # Save outputs
    print(f"\n[5/5] Saving outputs...")
    results.to_parquet(OUT_PARQUET, index=False, engine="pyarrow", compression="snappy")
    print(f"  ✓ Full results : {OUT_PARQUET}  ({OUT_PARQUET.stat().st_size/1024:.1f} KB)")

    output_json = format_output(results, TOP_N, us_date)
    with open(OUT_JSON, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  ✓ Top picks    : {OUT_JSON}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  INFERENCE COMPLETE")
    print(f"  US signal  : {us_date} | S&P={us_features.get('sp500_chg',0):+.2f}%  "
          f"VIX={us_features.get('vix_level',0):.1f}")
    print(f"  Top pick   : {results.iloc[0]['ticker']}  "
          f"(proba={results.iloc[0]['proba_up']:.3f})")
    print(f"  Total picks: {above} tickers above {INFERENCE_THRESH} threshold")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()