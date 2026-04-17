"""
train.py
=========
Training harian XGBoost classifier untuk prediksi IDX movers.

Logic:
  - US features tanggal T  →  IDX label tanggal T+1
  - Join: idx_ohlcv["date"] == us_features["date"] + 1 trading day
  - Label: next_day_ret > 3% → 1, else → 0
  - Handle class imbalance via scale_pos_weight
  - Save model ke ml_data/model.json (XGBoost native format)
  - Save feature list ke ml_data/feature_cols.json

Dijalankan: harian via GH Actions setelah append_daily.py
Output    : ml_data/model.json, ml_data/feature_cols.json, ml_data/train_report.json
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("ml_data")
IDX_FILE   = DATA_DIR / "idx_ohlcv.parquet"
US_FILE    = DATA_DIR / "us_features.parquet"
MODEL_FILE = DATA_DIR / "model.json"
FEAT_FILE  = DATA_DIR / "feature_cols.json"
REPORT_FILE= DATA_DIR / "train_report.json"

LABEL_THRESHOLD = 3.0   # % next day return threshold
ROLLING_WINDOW  = 504   # ~2 tahun trading days untuk rolling train (0 = pakai semua)
CV_SPLITS       = 5     # TimeSeriesSplit folds untuk evaluasi

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "eval_metric":      "aucpr",           # area under precision-recall, lebih baik untuk imbalanced
    "n_estimators":     300,
    "max_depth":        5,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,                # hindari overfitting di minority class
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",            # cepat untuk dataset besar
}

# US feature columns yang akan di-join
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

# IDX ticker-level feature columns
IDX_FEATURE_COLS = [
    "ret_1d", "ret_3d", "ret_5d", "ret_20d",
    "vol_ratio", "volatility_20d",
    "above_ma5", "above_ma20",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[1/5] Loading parquet data...")
    idx = pd.read_parquet(IDX_FILE)
    us  = pd.read_parquet(US_FILE)

    idx["date"] = pd.to_datetime(idx["date"])
    us.index    = pd.to_datetime(us.index)

    print(f"  IDX : {len(idx):,} rows, {idx['ticker'].nunique()} tickers")
    print(f"  US  : {len(us):,} trading days")
    return idx, us


def build_joined_dataset(idx: pd.DataFrame, us: pd.DataFrame) -> pd.DataFrame:
    """
    Join IDX dengan US features.

    Key insight: IDX row tanggal T menggunakan US features tanggal T-1
    (US close dini hari WIB → IDX open jam 9 WIB hari berikutnya).

    Cara join:
      1. Buat mapping: us_date → next_trading_date (T → T+1)
      2. Join idx.date == next_trading_date
    """
    print("[2/5] Building joined dataset...")

    # Sort US dates
    us_sorted = us.sort_index()
    us_dates  = us_sorted.index.tolist()

    # Mapping: tanggal US T → tanggal IDX yang akan terpengaruh (T+1 trading day)
    # Artinya: untuk IDX tanggal X, cari US tanggal sebelumnya
    us_date_series = pd.Series(us_dates, name="us_date")

    # Reset US index jadi column untuk merge
    us_reset = us_sorted[US_FEATURE_COLS].copy()
    us_reset.index.name = "us_date"
    us_reset = us_reset.reset_index()

    # Untuk setiap IDX date, kita perlu US date T-1
    # Strategi: merge_asof — untuk setiap IDX date, ambil US date terdekat sebelumnya
    idx_dates = idx[["date"]].drop_duplicates().sort_values("date")

    # merge_asof: untuk setiap idx date, cari us_date <= idx_date
    idx_dates = pd.merge_asof(
        idx_dates,
        us_reset[["us_date"]].rename(columns={"us_date": "us_date"}),
        left_on="date",
        right_on="us_date",
        direction="backward",
    )

    # Tapi kita mau US dari SEBELUM IDX date tersebut, bukan hari yang sama
    # Karena US close jam 4 ET = jam 4 pagi WIB = sebelum IDX open jam 9 WIB hari yg sama
    # Jadi US tanggal T BISA dipakai untuk IDX tanggal T (hari yang sama)
    # KECUALI jika IDX hari T == US hari T (overlap), maka sudah benar
    # → pakai direction="backward" sudah correct: ambil US date <= IDX date

    # Merge US features ke idx_dates
    idx_with_us_date = pd.merge(idx_dates, us_reset, left_on="us_date", right_on="us_date")

    # Join ke IDX full dataframe
    df = pd.merge(idx, idx_with_us_date, on="date", how="inner")

    before = len(df)
    df = df.dropna(subset=US_FEATURE_COLS + IDX_FEATURE_COLS + ["label"])
    print(f"  Joined rows  : {before:,}")
    print(f"  After dropna : {len(df):,}")
    print(f"  Date range   : {df['date'].min().date()} → {df['date'].max().date()}")

    # Label distribution
    dist = df["label"].value_counts(normalize=True) * 100
    print(f"  Label dist   : 0={dist.get(0,0):.1f}%  1={dist.get(1,0):.1f}%")

    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """Prepare X, y dan return feature column list."""
    print("[3/5] Preparing features...")

    feature_cols = US_FEATURE_COLS + IDX_FEATURE_COLS

    # Filter only available columns
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["label"].astype(int)

    # Fill any remaining NaN dengan median
    X = X.fillna(X.median())

    print(f"  Features     : {len(feature_cols)}")
    print(f"  Samples      : {len(X):,}")
    print(f"  Positive rate: {y.mean()*100:.1f}%")

    return X, y, feature_cols


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list,
    dates: pd.Series,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    Train XGBoost dengan TimeSeriesSplit CV untuk evaluasi,
    lalu retrain pada full dataset untuk model final.
    """
    print("[4/5] Training XGBoost...")

    # scale_pos_weight: handle class imbalance
    # ratio = jumlah negatif / jumlah positif
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    spw = neg_count / pos_count
    print(f"  scale_pos_weight = {spw:.2f}  ({neg_count:,} neg / {pos_count:,} pos)")

    params = {**XGB_PARAMS, "scale_pos_weight": spw}

    # ── TimeSeriesSplit CV untuk evaluasi ─────────────────────────────────────
    print(f"\n  Running {CV_SPLITS}-fold TimeSeriesSplit CV...")
    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)

    # Sort by date untuk time series split yang benar
    sort_idx = dates.argsort()
    X_sorted = X.iloc[sort_idx].reset_index(drop=True)
    y_sorted = y.iloc[sort_idx].reset_index(drop=True)

    cv_metrics = []
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
        X_tr, X_val = X_sorted.iloc[train_idx], X_sorted.iloc[val_idx]
        y_tr, y_val = y_sorted.iloc[train_idx], y_sorted.iloc[val_idx]

        clf = xgb.XGBClassifier(**params, verbosity=0)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        # Gunakan threshold lebih tinggi untuk precision (kita mau confident picks)
        y_pred_strict = (y_prob >= 0.6).astype(int)

        auc     = roc_auc_score(y_val, y_prob)
        prec    = precision_score(y_val, y_pred_strict, zero_division=0)
        recall  = recall_score(y_val, y_pred_strict, zero_division=0)

        cv_metrics.append({"fold": fold, "auc": auc, "precision": prec, "recall": recall})
        print(f"  Fold {fold}: AUC={auc:.3f}  Prec={prec:.3f}  Recall={recall:.3f}  "
              f"(val_size={len(val_idx):,})")

    avg_auc  = np.mean([m["auc"]       for m in cv_metrics])
    avg_prec = np.mean([m["precision"] for m in cv_metrics])
    avg_rec  = np.mean([m["recall"]    for m in cv_metrics])
    print(f"\n  CV Average: AUC={avg_auc:.3f}  Prec={avg_prec:.3f}  Recall={avg_rec:.3f}")

    # ── Final model: train pada full dataset ──────────────────────────────────
    print("\n  Training final model on full dataset...")

    # Jika ROLLING_WINDOW > 0, hanya pakai N hari terakhir
    if ROLLING_WINDOW > 0:
        unique_dates = sorted(dates.unique())
        if len(unique_dates) > ROLLING_WINDOW:
            cutoff_date = unique_dates[-ROLLING_WINDOW]
            mask = dates >= cutoff_date
            X_final = X[mask]
            y_final = y[mask]
            print(f"  Rolling window: using last {ROLLING_WINDOW} trading days "
                  f"({X_final['ret_1d'].count():,} rows)")
        else:
            X_final, y_final = X, y
    else:
        X_final, y_final = X, y

    final_clf = xgb.XGBClassifier(**params, verbosity=0)
    final_clf.fit(X_final, y_final, verbose=False)

    # Feature importance
    importance = dict(zip(feature_cols, final_clf.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    print("\n  Top 10 feature importances:")
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10], 1):
        bar = "█" * int(imp * 200)
        print(f"    {i:2d}. {feat:<20} {imp:.4f}  {bar}")

    report = {
        "cv_folds":     CV_SPLITS,
        "avg_auc":      round(avg_auc, 4),
        "avg_precision":round(avg_prec, 4),
        "avg_recall":   round(avg_rec, 4),
        "cv_metrics":   cv_metrics,
        "scale_pos_weight": round(spw, 2),
        "n_train_rows": len(X_final),
        "n_features":   len(feature_cols),
        "feature_importance": {k: round(v, 4) for k, v in importance_sorted.items()},
        "label_threshold_pct": LABEL_THRESHOLD,
        "inference_threshold": 0.6,
    }

    return final_clf, report


def save_outputs(
    model: xgb.XGBClassifier,
    feature_cols: list,
    report: dict,
) -> None:
    print("[5/5] Saving outputs...")

    # Model
    model.save_model(MODEL_FILE)
    print(f"  ✓ Model    : {MODEL_FILE}  ({MODEL_FILE.stat().st_size/1024:.1f} KB)")

    # Feature list (penting: inference.py harus pakai urutan yang sama persis)
    with open(FEAT_FILE, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  ✓ Features : {FEAT_FILE}")

    # Training report
    from datetime import datetime, timezone
    report["trained_at"] = datetime.now(timezone.utc).isoformat()

    # Convert numpy types → Python native (float32/int64 not JSON serializable)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    report = convert_numpy(report)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Report   : {REPORT_FILE}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"{'='*60}")
    print(f"  Quant Radar — IDX ML Training")
    print(f"  Model  : XGBoost Classifier")
    print(f"  Label  : next_day_ret > {LABEL_THRESHOLD}%")
    print(f"{'='*60}\n")

    idx, us   = load_data()
    df        = build_joined_dataset(idx, us)
    X, y, fc  = prepare_features(df)
    model, rpt= train_model(X, y, fc, df["date"])
    save_outputs(model, fc, rpt)

    print(f"\n{'='*60}")
    print(f"  DONE — Model ready for inference.py")
    print(f"  AUC={rpt['avg_auc']:.3f}  "
          f"Prec={rpt['avg_precision']:.3f}  "
          f"Recall={rpt['avg_recall']:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
