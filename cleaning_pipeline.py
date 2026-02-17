import pandas as pd
import numpy as np
from pathlib import Path

# ========================
# CONFIG (ROBUST PATH)
# ========================

BASE_DIR = Path(__file__).resolve().parent
RAW_FOLDER = BASE_DIR / "raw_daily"
OUTPUT_FILE = BASE_DIR / "master.parquet"
PROCESSED_LOG = BASE_DIR / "processed_files.txt"

# ========================
# COLUMN RENAME MAP
# ========================

COLUMN_MAP = {
    "Kode Saham": "ticker",
    "Tanggal Perdagangan Terakhir": "date",
    "Open Price": "open",
    "Tertinggi": "high",
    "Terendah": "low",
    "Penutupan": "close",
    "Volume": "volume",
    "Nilai": "value",
    "Frekuensi": "freq",
    "Foreign Buy": "foreign_buy",
    "Foreign Sell": "foreign_sell",
    "Offer Volume": "offer_volume",
    "Bid Volume": "bid_volume",
    "Non Regular Volume": "non_regular_volume",
    "Non Regular Value": "non_regular_value",
    "Non Regular Frequency": "non_regular_freq",
}

# ========================
# HELPER FUNCTIONS
# ========================

def clean_number(series):
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("-", "", regex=False)
        .replace("", np.nan)
        .astype(float)
    )

# ========================
# MASTER SANITY CHECK
# ========================

def master_sanity_check(master):
    print("\n=== MASTER SANITY CHECK ===")

    issues = False

    # duplicate check
    dup = master.duplicated(subset=["ticker", "date"]).sum()
    print("Duplicate ticker-date rows:", dup)
    if dup > 0:
        issues = True

    # missing values
    print("\nMissing values:")
    print(master.isna().sum())

    # invalid price
    if "close" in master.columns:
        bad_price = (master["close"] <= 0).sum()
        print("\nInvalid price rows:", bad_price)
        if bad_price > 0:
            issues = True

    # invalid volume
    if "volume" in master.columns:
        bad_vol = (master["volume"] < 0).sum()
        print("Invalid volume rows:", bad_vol)
        if bad_vol > 0:
            issues = True

    # check date order per ticker
    bad_order = 0
    for _, g in master.groupby("ticker"):
        if not g["date"].is_monotonic_increasing:
            bad_order += 1

    print("Tickers with unordered dates:", bad_order)
    if bad_order > 0:
        issues = True

    # optional hard stop
    if issues:
        print("\n⚠️ WARNING: Data quality issues detected")
        # uncomment kalau mau pipeline auto stop
        # raise Exception("Master dataset failed sanity check")

    else:
        print("\n✅ Master dataset looks healthy")


# ========================
# SANITY CHECK
# ========================

print("\n=== SANITY CHECK ===")
print("Script location:", BASE_DIR)
print("Raw folder exists:", RAW_FOLDER.exists())

if not RAW_FOLDER.exists():
    raise Exception("Folder raw_daily tidak ditemukan.")

files = list(RAW_FOLDER.glob("*.xlsx")) + list(RAW_FOLDER.glob("*.csv"))
print("Excel/CSV files found:", len(files))

if len(files) == 0:
    raise Exception("Tidak ada file .xlsx / .csv")

# ========================
# LOAD PROCESSED FILE LIST
# ========================

if PROCESSED_LOG.exists():
    processed = set(PROCESSED_LOG.read_text().splitlines())
else:
    processed = set()

print("Already processed:", len(processed))

# ========================
# LOAD ONLY NEW FILES
# ========================

dfs = []
new_processed = []

for file in files:

    if file.name in processed:
        continue

    print("\nProcessing NEW:", file.name)

    try:
        if file.suffix == ".xlsx":
            df = pd.read_excel(file)
        elif file.suffix == ".csv":
            df = pd.read_csv(file)
        else:
            continue

        # rename
        df = df.rename(columns=COLUMN_MAP)

        keep_cols = list(COLUMN_MAP.values())
        df = df[[c for c in keep_cols if c in df.columns]]

        # clean ticker
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.strip()

        # convert date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # numeric convert
        numeric_cols = [
            "open","high","low","close",
            "volume","value","freq",
            "foreign_buy","foreign_sell",
            "offer_volume","bid_volume",
            "non_regular_volume","non_regular_value","non_regular_freq"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = clean_number(df[col])

        dfs.append(df)
        new_processed.append(file.name)

    except Exception as e:
        print("⚠️ Skip:", file.name, e)

# ========================
# STOP IF NO NEW DATA
# ========================

if len(dfs) == 0:
    print("\n✅ No new files to process.")
    exit()

print("\nMerging new data...")
new_data = pd.concat(dfs, ignore_index=True)

# ========================
# LOAD OLD MASTER (IF EXISTS)
# ========================

if OUTPUT_FILE.exists():
    print("Loading existing master...")
    old_master = pd.read_parquet(OUTPUT_FILE)
    master = pd.concat([old_master, new_data], ignore_index=True)
else:
    print("Creating new master dataset...")
    master = new_data

# ========================
# BASIC CLEANING
# ========================

print("Cleaning master dataset...")

master = master.dropna(subset=["ticker", "date"])
master = master.drop_duplicates(subset=["ticker", "date"])
master = master.sort_values(["ticker", "date"])

optional_cols = [
    "foreign_buy","foreign_sell",
    "offer_volume","bid_volume",
    "non_regular_volume","non_regular_value","non_regular_freq"
]

for col in optional_cols:
    if col in master.columns:
        master[col] = master[col].fillna(0)

master = master.reset_index(drop=True)

# ========================
# VALIDATION
# ========================

print("\n=== DATA SUMMARY ===")
print("Rows:", len(master))
print("Unique tickers:", master["ticker"].nunique())
print("Date range:", master["date"].min(), "→", master["date"].max())

# run full sanity check
master_sanity_check(master)

# ========================
# SAVE MASTER
# ========================

print("\nSaving master dataset...")
master.to_parquet(OUTPUT_FILE, index=False)

# save processed file list
processed.update(new_processed)
PROCESSED_LOG.write_text("\n".join(processed))

print("\n✅ DONE")
print("New files processed:", len(new_processed))
